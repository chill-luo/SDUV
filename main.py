import os
import re
import time
import yaml
import torch
import datetime

import numpy as np
import torch.nn as nn

from skimage import io
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import trainset, testset, train_preprocess, test_preprocess, inv_normalize
from network import UNet3D
from utils import QueueList, XlsBook


def train(args):
    exp_name = '_'.join((args.datasets_name, datetime.datetime.now().strftime("%Y%m%d%H%M")))
    save_path = os.path.join(args.pth_dir, exp_name)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(args, f)

    local_model = UNet3D(in_channels=1,
                         out_channels=1,
                         f_maps=args.fmap,
                         final_sigmoid=True)
    name_list, coordinate_list, stack_index, noise_im_all = train_preprocess(args)
    optimizer_G = torch.optim.Adam(local_model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    train_data = trainset(name_list, coordinate_list, noise_im_all, stack_index)
    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    ngpu = str(args.GPU).count(',') + 1
    if torch.cuda.is_available():
        local_model = local_model.cuda()
        local_model = nn.DataParallel(local_model, device_ids=range(ngpu))

    prev_time = time.time()
    time_queue = QueueList(max_size=25)
    test_time = 0
    train_logger = XlsBook(['Epoch', 'Iteration', 'Loss_rec', 'Loss_reg', 'Loss_total'])
    L1_pixelwise = torch.nn.L1Loss().cuda()
    L2_pixelwise = torch.nn.MSELoss().cuda()

    print('\033[1;31m===== Start Training ===== \033[0m')
    for epoch in range(0, args.n_epochs):
        with tqdm(total=len(trainloader), desc=f'Epoch {epoch + 1}/{args.n_epochs}', unit='batch') as pbar:
            for iteration, patch in enumerate(trainloader):
                patch = patch.cuda()
                depth = patch.shape[2]
                sub1 = patch[:, :, 0:depth:2, :, :]
                sub2 = patch[:, :, 1:depth:2, :, :]
                
                patch_outp = local_model(patch).detach()
                sub_outp = local_model(sub1)

                rec_loss = 0.5 * L1_pixelwise(sub_outp, sub2) + 0.5 * L2_pixelwise(sub_outp, sub2)
                reg_loss = L2_pixelwise(sub_outp-patch_outp[:, :, 0:depth:2, :, :], sub2-patch_outp[:, :, 1:depth:2, :, :])
                loss = rec_loss + reg_loss * args.gama

                optimizer_G.zero_grad()
                loss.backward()
                optimizer_G.step()

                epochs_left = args.n_epochs - epoch
                batches_done = epoch * len(trainloader) + iteration
                batches_left = args.n_epochs * len(trainloader) - batches_done
                iter_time = time.time() - prev_time
                if iteration == 0:
                    iter_time -= test_time
                time_queue.add(iter_time)
                iter_time_avg = np.array(time_queue.list).mean()
                time_left = datetime.timedelta(seconds=int(batches_left * iter_time_avg + epochs_left * test_time))
                prev_time = time.time()

                pbar.set_postfix(**{'loss': loss.item(), 'ETA': time_left})
                train_logger.write([epoch+1, iteration+1, rec_loss.item(), reg_loss.item(), loss.item()])
                pbar.update(1)

        if (epoch + 1) % args.save_round == 0:
            model_save_name = 'Epoch_' + str(epoch + 1).zfill(3) + '.pth'
            model_save_path = os.path.join(save_path, model_save_name)

            if isinstance(local_model, nn.DataParallel):
                torch.save(local_model.module.state_dict(), model_save_path)
            else:
                torch.save(local_model.state_dict(), model_save_path)
            print(f'Model {model_save_name} saved.')

        if args.validation:
            print(f'Validation for Epoch {epoch + 1}:')
            test_prev_time = time.time()

            validate(args, local_model, epoch, save_path)

            test_time = time.time() - test_prev_time
            print()
    
    if args.train_log:
        train_logger.save(os.path.join(save_path, 'logger.xlsx'))
    print(f"Train of '{args.notes}' finished. Models have been saved to '{save_path}'.")


def validate(args, local_model, train_epoch, pth_path):
    name_list, noise_img, coordinate_list, test_im_name, norm_param, input_data_type = test_preprocess(args, img_id=0)
    denoise_img = np.zeros(noise_img.shape)
    input_img = np.zeros(noise_img.shape)
    test_data = testset(name_list, coordinate_list, noise_img)
    testloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)

    with tqdm(total=len(testloader), desc=f'Epoch {train_epoch + 1}', unit='patch') as pbar:
        for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
            noise_patch = noise_patch.cuda()
            real_A = noise_patch
            real_A = Variable(real_A)
            fake_B = local_model(real_A)

            output_image = np.squeeze(fake_B.cpu().detach().numpy())
            raw_image = np.squeeze(real_A.cpu().detach().numpy())
            assert output_image.ndim == 3

            stack_start_w = int(single_coordinate['stack_start_w'])
            stack_end_w = int(single_coordinate['stack_end_w'])
            patch_start_w = int(single_coordinate['patch_start_w'])
            patch_end_w = int(single_coordinate['patch_end_w'])

            stack_start_h = int(single_coordinate['stack_start_h'])
            stack_end_h = int(single_coordinate['stack_end_h'])
            patch_start_h = int(single_coordinate['patch_start_h'])
            patch_end_h = int(single_coordinate['patch_end_h'])

            stack_start_s = int(single_coordinate['stack_start_s'])
            stack_end_s = int(single_coordinate['stack_end_s'])
            patch_start_s = int(single_coordinate['patch_start_s'])
            patch_end_s = int(single_coordinate['patch_end_s'])

            output_patch = output_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
            raw_patch = raw_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]

            output_patch = inv_normalize(output_patch, norm_param, mode=args.norm_mode)
            raw_patch = inv_normalize(raw_patch, norm_param, mode=args.norm_mode)

            scale_factor = np.sum(raw_patch) / np.sum(output_patch)
            denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                = output_patch * scale_factor ** 0.5
            input_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                = raw_patch
            
            pbar.update(1)

    input_img = input_img.squeeze().astype(input_data_type)
    output_img = denoise_img.squeeze().astype(np.float32)
    del denoise_img

    if input_data_type == 'uint16':
        output_img=np.clip(output_img, 0, 65535)
        output_img = output_img.astype('uint16')
    elif input_data_type == 'int16':
        output_img=np.clip(output_img, -32767, 32767)
        output_img = output_img.astype('int16')
    else:
        output_img = output_img.astype('int32')
    
    if args.val_datasize < output_img.shape[0]:
        test_start = (output_img.shape[0] - args.val_datasize) // 2
        test_stop = test_start + args.val_datasize
        output_img = output_img[test_start:test_stop, :, :]

    os.makedirs(os.path.join(pth_path, 'val'), exist_ok=True)
    result_name = '_'.join((os.path.splitext(test_im_name)[0], 'Epoch', str(train_epoch + 1).zfill(3) + '.tif'))
    result_path = os.path.join(pth_path, 'val', result_name)
    io.imsave(result_path, output_img, check_contrast=False)
    

def test(args):
    exp_name = '_'.join(('DataFolderIs', args.datasets_name, 'ModelFolderIs', args.denoise_model))
    output_path = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f'config_{datetime.datetime.now().strftime("%Y%m%d%H%M")}.yaml'), 'w') as f:
        yaml.dump(args, f)

    pths_num = [int(p) for p in args.pths_num.split(',')]
    model_path = os.path.join(args.pth_dir, args.denoise_model)
    model_list = [m for m in os.listdir(model_path) if ('.pth' in m) and (int(re.split(r'[_.]', m)[1]) in pths_num)]
    model_list.sort()

    im_folder = args.datasets_path
    img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
    img_list.sort()
    print('\033[1;31m===== Stack List ===== \033[0m')
    for img in enumerate(img_list): 
        print(img)
    print()

    denoise_generator = UNet3D(in_channels=1,
                                out_channels=1,
                                f_maps=args.fmap,
                                final_sigmoid=True)
    local_model = denoise_generator

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    ngpu = str(args.GPU).count(',') + 1
    if torch.cuda.is_available():
        local_model = local_model.cuda()
        local_model = nn.DataParallel(local_model, device_ids=range(ngpu))

    start_time = time.time()
    print('\033[1;31m===== Start Testing ===== \033[0m')
    for pth_index in range(len(model_list)):
        pth_name = model_list[pth_index]
        output_path_name = os.path.join(output_path, os.path.splitext(pth_name)[0])
        os.makedirs(output_path_name, exist_ok=True)

        model_name = os.path.join(args.pth_dir, args.denoise_model, pth_name)
        if isinstance(local_model, nn.DataParallel):
            local_model.module.load_state_dict(torch.load(model_name))
            local_model.eval()
        else:
            local_model.load_state_dict(torch.load(model_name))
            local_model.eval()
        local_model.cuda()

        for N in range(len(img_list)):
            name_list, noise_img, coordinate_list,test_im_name, norm_param, input_data_type = test_preprocess(args, N)
            denoise_img = np.zeros(noise_img.shape)
            input_img = np.zeros(noise_img.shape)

            test_data = testset(name_list, coordinate_list, noise_img)
            testloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
            
            print(f'Model information: {pth_name}')
            with tqdm(total=len(testloader), desc=f'[Model {pth_index+1}/{len(model_list)}, Stack {N + 1}/{len(img_list)}]', unit='patch') as pbar:
                for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
                    noise_patch = noise_patch.cuda()
                    real_A = noise_patch

                    with torch.no_grad():
                        fake_B = local_model(real_A)

                    output_image = np.squeeze(fake_B.cpu().detach().numpy())
                    raw_image = np.squeeze(real_A.cpu().detach().numpy())
                    assert output_image.ndim == 3

                    stack_start_w = int(single_coordinate['stack_start_w'])
                    stack_end_w = int(single_coordinate['stack_end_w'])
                    patch_start_w = int(single_coordinate['patch_start_w'])
                    patch_end_w = int(single_coordinate['patch_end_w'])

                    stack_start_h = int(single_coordinate['stack_start_h'])
                    stack_end_h = int(single_coordinate['stack_end_h'])
                    patch_start_h = int(single_coordinate['patch_start_h'])
                    patch_end_h = int(single_coordinate['patch_end_h'])

                    stack_start_s = int(single_coordinate['stack_start_s'])
                    stack_end_s = int(single_coordinate['stack_end_s'])
                    patch_start_s = int(single_coordinate['patch_start_s'])
                    patch_end_s = int(single_coordinate['patch_end_s'])

                    output_patch = output_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]
                    raw_patch = raw_image[patch_start_s:patch_end_s, patch_start_h:patch_end_h, patch_start_w:patch_end_w]

                    output_patch = inv_normalize(output_patch, norm_param, mode=args.norm_mode)
                    raw_patch = inv_normalize(raw_patch, norm_param, mode=args.norm_mode)

                    scale_factor = np.sum(raw_patch) / np.sum(output_patch)
                    denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                        = output_patch * scale_factor ** 0.5
                    input_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                        = raw_patch
                    
                    pbar.update(1)

            input_img = input_img.squeeze().astype(input_data_type)
            output_img = denoise_img.squeeze().astype(np.float32)
            del denoise_img
                
            if input_data_type == 'uint16':
                output_img=np.clip(output_img, 0, 65535)
                output_img = output_img.astype('uint16')
            elif input_data_type == 'int16':
                output_img=np.clip(output_img, -32767, 32767)
                output_img = output_img.astype('int16')
            else:
                output_img = output_img.astype('int32')

            result_name = '_'.join((os.path.splitext(test_im_name)[0], os.path.splitext(pth_name)[0], 'output.tif'))
            result_path = os.path.join(output_path_name, result_name)
            io.imsave(result_path, output_img, check_contrast=False)

            print()

    stop_time = time.time()
    print(f"All finished. Results have been saved to '{output_path}'.")
    print(f'Time used: {datetime.timedelta(seconds=int(stop_time - start_time))}')