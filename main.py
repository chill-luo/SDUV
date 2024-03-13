import os
import time
import torch
import datetime

import numpy as np
import torch.nn as nn
import tifffile as tiff

from skimage import io
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import trainset, testset, train_preprocess_lessMemoryMulStacks, test_preprocess_chooseOne, inv_normalize
from network import UNet3D
# from evaluate import compare, general_evaluate
from utils import QueueList, XlsBook


def train(args):
    exp_name = '_'.join(args.datasets_name, datetime.datetime.now().strftime("%Y%m%d%H%M"))
    save_path = os.path.join(args.pth_dir, exp_name)
    os.makedirs(save_path, exist_ok=True)

    local_model = UNet3D(in_channels=1,
                         out_channels=1,
                         f_maps=args.fmap,
                         final_sigmoid=True)
    name_list, coordinate_list, stack_index, noise_im_all = train_preprocess_lessMemoryMulStacks(args)
    optimizer_G = torch.optim.Adam(local_model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    train_data = trainset(name_list, coordinate_list, noise_im_all, stack_index)
    trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    ngpu = str(args.GPU).count(',') + 1
    if torch.cuda.is_available():
        local_model = local_model.cuda()
        local_model = nn.DataParallel(local_model, device_ids=range(ngpu))
        print('\033[1;31mUsing {} GPU(s) for training -----> \033[0m'.format(torch.cuda.device_count()))

    # train
    prev_time = time.time()
    # time_start = time.time()
    time_queue = QueueList(max_size=25)
    test_time = 0
    train_logger = XlsBook(['Epoch', 'Iteration', 'L1', 'L2', 'Loss'])
    L1_pixelwise = torch.nn.L1Loss().cuda()
    L2_pixelwise = torch.nn.MSELoss().cuda()

    for epoch in range(0, args.n_epochs):
        # train_data = trainset(name_list, coordinate_list, noise_im_all, stack_index)
        # trainloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        with tqdm(total=len(trainloader), desc=f'Epoch {epoch + 1}/{args.n_epochs}', unit='patch') as pbar:
            for iteration, patch in enumerate(trainloader):
                # The input volume and corresponding target volume from data loader to train the deep neural network
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

                # Record and estimate the remaining time
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
                # if iteration % 1 == 0:
                #     time_end = time.time()
                #     print(
                #         '\r[Epoch %d/%d] [Batch %d/%d] [Total loss: %.2f, REC Loss: %.2f, REG Loss: %.2f] [ETA: %s] [Time cost: %.2d s]     '
                #         % (
                #             epoch + 1,
                #             args.n_epochs,
                #             iteration + 1,
                #             len(trainloader),
                #             loss.item(),
                #             rec_loss.item(),
                #             reg_loss.item(),
                #             time_left,
                #             time_end - time_start
                #         ), end=' ')
                #     train_logger.write([epoch+1, iteration+1, rec_loss.item(), reg_loss.item(), loss.item()])
                pbar.update(args.batch_size)

        if args.validation:
            print(f'\nValidating model of epoch {epoch + 1} on the first noisy file ----->')
            test_prev_time = time.time()

            validate(args, local_model, epoch, iteration, save_path)

            test_time = time.time() - test_prev_time
            print('\n', end=' ')


        if (epoch + 1) % args.save_round == 0:
            model_save_name = 'Epoch_' + str(epoch + 1).zfill(3) + '.pth'
            model_save_path = os.path.join(save_path, model_save_name)

            if isinstance(local_model, nn.DataParallel):
                torch.save(local_model.module.state_dict(), model_save_path)  # parallel
            else:
                torch.save(local_model.state_dict(), model_save_path)  # not parallel

            print(f'Model of epoch {epoch + 1} has been saved as {model_save_path}')
    
    train_logger.save(os.path.join(save_path, 'logger_train.xlsx'))
    print(f"Train of '{args.notes}' finished. Save all models to '{save_path}'.")


def validate(args, local_model, train_epoch, train_iteration, pth_path):
    """
    Pytorch testing workflow
    Args:
        train_epoch : current train epoch number
        train_iteration : current train_iteration number
    """
    # Crop test file into 3D patches for inference
    name_list, noise_img, coordinate_list, test_im_name, norm_param, input_data_type = test_preprocess_chooseOne(args, img_id=0)
    # Record the inference time
    prev_time = time.time()
    # time_start = time.time()
    denoise_img = np.zeros(noise_img.shape)
    input_img = np.zeros(noise_img.shape)
    test_data = testset(name_list, coordinate_list, noise_img)
    testloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # test_logger = XlsBook(['Epoch', 'SSIM', 'PSNR'])
    # gt_image = None
    # ssim_ref = None
    # psnr_ref = None

    with tqdm(total=len(testloader), desc=f'Epoch {train_epoch + 1}', unit='patch') as pbar:
        for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
            # Pre-trained models are loaded into memory and the sub-stacks are directly fed into the model.
            noise_patch = noise_patch.cuda()
            real_A = noise_patch
            real_A = Variable(real_A)
            fake_B = local_model(real_A)
            
            batches_done = iteration
            batches_left = 1 * len(testloader) - batches_done
            time_left_seconds = int(batches_left * (time.time() - prev_time))
            # time_left = datetime.timedelta(seconds=time_left_seconds)
            prev_time = time.time()
            # time_end = time.time()
            # time_cost = time_end - time_start  # datetime.timedelta(seconds= (time_end - time_start))
            pbar.set_postfix(**{'Time left': f'{time_left_seconds} s'})
            # print(
            #     '\r [Patch %d/%d] [Time Cost: %.0d s] [ETA: %s s]     '
            #     % (
            #         iteration + 1,
            #         len(testloader),
            #         time_cost,
            #         time_left_seconds
            #     ), end=' ')

            # if (iteration + 1) % len(testloader) == 0:
            #     print('\n', end=' ')

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

    # Stitching finish
    input_img = input_img.squeeze().astype(input_data_type)
    output_img = denoise_img.squeeze().astype(np.float32)
    del denoise_img

    # Evaluate denoised image
    if input_data_type == 'uint16':
        output_img=np.clip(output_img, 0, 65535)
        output_img = output_img.astype('uint16')
    elif input_data_type == 'int16':
        output_img=np.clip(output_img, -32767, 32767)
        output_img = output_img.astype('int16')
    else:
        output_img = output_img.astype('int32')
    
    # gt_image = tiff.imread(os.path.join(args.gt_path, test_im_name))
    # ssim_ref, psnr_ref = compare(input_img, gt_image)
    # ssim, psnr = compare(output_img, gt_image)
    # print('Evaluation results of denoised images ---> SSIM: %.5f(%.5f) & PSNR: %.4f(%.4f)' % (ssim, ssim_ref, psnr, psnr_ref))
    # test_logger.write([train_epoch+1, ssim, psnr])

    # Save inference image
    # if (train_epoch + 1) % 5 == 0:
    if args.val_datasize < output_img.shape[0]:
        test_start = (output_img.shape[0] - args.val_datasize) // 2
        test_stop = test_start + args.val_datasize
        output_img = output_img[test_start:test_stop, :, :]

    result_name = os.path.splitext(test_im_name)[0] + '_' + 'Epoch_' + str(train_epoch + 1).zfill(3) + '.tif'
    result_path = os.path.join(pth_path, result_name)
    io.imsave(result_path, output_img, check_contrast=False)
    

def test(args):
    """
    Pytorch testing workflow

    """
    # prepare
    # if args.datasets_path[-1]!='/':
    #     datasets_name=args.datasets_path.split("/")[-1]
    # else:
    #     datasets_name=args.datasets_path.split("/")[-2]
    # self.datasets_name = self.datasets_path.split("/", 1)[1]
    # pth_name = self.datasets_name + '_' + datetime.datetime.now().strftime("%Y%m%d%H%M")
    # self.pth_path = self.pth_dir + '/' + pth_name
    # if not os.path.exists(self.pth_path):
    #     os.makedirs(self.pth_path)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    # current_time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    exp_name = 'DataFolderIs_' + args.datasets_name + '-ModelFolderIs_' + args.denoise_model
    output_path = os.path.join(args.output_dir, exp_name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # if args.ref_metrics:
    #     logger = XlsBook(['Model', 'Image', 'SSIM', 'SSIM_ref', 'PSNR', 'PSNR_ref'])
    # else:
    #     logger = XlsBook(['Model', 'Image', 'SSIM', 'PSNR'])
    
    # read model and image
    model_path = os.path.join(args.pth_dir, args.denoise_model)
    model_list = list(os.walk(model_path, topdown=False))[-1][-1]
    model_list.sort()

    # count_pth = 0
    # for i in range(len(model_list)):
    #     aaa = model_list[i]
    #     if '.pth' in aaa:
    #         count_pth = count_pth + 1
    # model_list_length = count_pth
    pths_num = [int(p) for p in args.pths_num.split(',')]

    im_folder = args.datasets_path
    img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
    img_list.sort()
    print('\033[1;31mStacks for processing -----> \033[0m')
    print('Total stack number -----> ', len(img_list))
    for img in img_list: 
        print(img)

    # init net
    denoise_generator = UNet3D(in_channels=1,
                                out_channels=1,
                                f_maps=args.fmap,
                                final_sigmoid=True)
    local_model = denoise_generator

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
    if torch.cuda.is_available():
        local_model = local_model.cuda()
        local_model = nn.DataParallel(local_model, device_ids=range(ngpu))
        print('\033[1;31mUsing {} GPU(s) for testing -----> \033[0m'.format(torch.cuda.device_count()))
    # cuda = True if torch.cuda.is_available() else False
    # Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    ngpu = str(args.GPU).count(',') + 1

    # Test
    pth_count=0
    for pth_index in range(len(model_list)):
        aaa = model_list[pth_index]
        if ('.pth' in aaa) and (int(aaa.split('_')[1]) in pths_num):
            pth_count=pth_count+1
            pth_name = model_list[pth_index]
            output_path_name = output_path + '//' + pth_name.replace('.pth', '')
            if not os.path.exists(output_path_name):
                os.mkdir(output_path_name)

            # load model
            model_name = args.pth_dir + '//' + args.denoise_model + '//' + pth_name
            if isinstance(local_model, nn.DataParallel):
                local_model.module.load_state_dict(torch.load(model_name))  # parallel
                local_model.eval()
            else:
                local_model.load_state_dict(torch.load(model_name))  # not parallel
                local_model.eval()
            local_model.cuda()
            # print_img_name = False
            # test all stacks
            # psnr_list = []
            # ssim_list = []
            for N in range(len(img_list)):
                name_list, noise_img, coordinate_list,test_im_name, norm_param, input_data_type = test_preprocess_chooseOne(args, N)
                # print(len(name_list))
                prev_time = time.time()
                # time_start = time.time()
                denoise_img = np.zeros(noise_img.shape)
                input_img = np.zeros(noise_img.shape)

                test_data = testset(name_list, coordinate_list, noise_img)
                testloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.num_workers)
                
                print(f'\nDenoising stack {os.path.splitext(test_im_name)[0]} with model {pth_name}:')
                with tqdm(total=len(testloader), desc=f'[Model {pth_count}/{len(pths_num)}, Stack {N + 1}/{len(img_list)}]', unit='patch') as pbar:
                    for iteration, (noise_patch, single_coordinate) in enumerate(testloader):
                        noise_patch = noise_patch.cuda()
                        real_A = noise_patch

                        with torch.no_grad():
                            fake_B = local_model(real_A)

                        # Determine approximate time left
                        batches_done = iteration
                        batches_left = 1 * len(testloader) - batches_done
                        time_left_seconds = int(batches_left * (time.time() - prev_time))
                        time_left = datetime.timedelta(seconds=time_left_seconds)
                        prev_time = time.time()

                        pbar.set_postfix(**{'Time left': f'{time_left_seconds} s'})
                        # if iteration % 1 == 0:
                        #     time_end = time.time()
                        #     time_cost = time_end - time_start  # datetime.timedelta(seconds= (time_end - time_start))
                        #     print(
                        #         '\r[Model %d/%d, %s] [Stack %d/%d, %s] [Patch %d/%d] [Time Cost: %.0d s] [ETA: %s s]     '
                        #         % (
                        #             pth_count,
                        #             len(pths_num),
                        #             pth_name,
                        #             N + 1,
                        #             len(img_list),
                        #             test_im_name.replace('.tif', ''),
                        #             iteration + 1,
                        #             len(testloader),
                        #             time_cost,
                        #             time_left_seconds
                        #         ), end=' ')

                        # if (iteration + 1) % len(testloader) == 0:
                        #     print('\n', end=' ')

                        # Enhanced sub-stacks are sequentially output from the network
                        output_image = np.squeeze(fake_B.cpu().detach().numpy())
                        raw_image = np.squeeze(real_A.cpu().detach().numpy())
                        assert output_image.ndim == 3

                        # The final enhanced stack can be obtained by stitching all sub-stacks.
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

                        # scale_factor = np.sum(raw_patch) / np.sum(np.clip(output_patch, 0, None))
                        scale_factor = np.sum(raw_patch) / np.sum(output_patch)
                        # scale_factor = 1
                        # if scale_factor <= 0:
                            # ipdb.set_trace()
                        # if scale_factor < 0:
                        #     scale_factor = 1
                        denoise_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                            = output_patch * scale_factor ** 0.5
                        input_img[stack_start_s:stack_end_s, stack_start_h:stack_end_h, stack_start_w:stack_end_w] \
                            = raw_patch
                        
                        pbar.update(1)

                # Stitching finish
                input_img = input_img.squeeze().astype(input_data_type)
                output_img = denoise_img.squeeze().astype(np.float32)
                del denoise_img
                    
                # Evaluate denoised image
                if input_data_type == 'uint16':
                    output_img=np.clip(output_img, 0, 65535)
                    output_img = output_img.astype('uint16')
                elif input_data_type == 'int16':
                    output_img=np.clip(output_img, -32767, 32767)
                    output_img = output_img.astype('int16')
                else:
                    output_img = output_img.astype('int32')

                # if args.evaluation:
                #     gt_image = tiff.imread(os.path.join(args.gt_path, test_im_name))
                #     # ssim_ref, psnr_ref = compare(input_img, gt_image)
                #     # ssim, psnr = compare(output_img.astype(input_data_type), gt_image)
                #     if args.ref_metrics:
                #         ssim_ref, psnr_ref = general_evaluate(input_img, gt_image, mode=args.evaluation)
                #     ssim, psnr = general_evaluate(output_img.astype(input_data_type), gt_image, mode=args.evaluation)
                #     if args.ref_metrics:
                #         print('Evaluation results of denoised images ---> SSIM: %.5f(%.5f) & PSNR: %.4f(%.4f)\n' % (ssim, ssim_ref, psnr, psnr_ref))
                #         logger.write([pth_name, test_im_name, ssim, ssim_ref, psnr, psnr_ref])
                #     else:
                #         print('Evaluation results of denoised images ---> SSIM: %.5f & PSNR: %.4f\n' % (ssim, psnr))
                #         logger.write([pth_name, test_im_name, ssim, psnr])
                #     ssim_list.append(ssim)
                #     psnr_list.append(psnr)

                # Save inference image
                result_name = test_im_name.replace('.tif','') + '_' + os.path.splitext(pth_name)[0] + '_output.tif'
                result_path = os.path.join(output_path_name, result_name)
                io.imsave(result_path, output_img, check_contrast=False)

            # if args.evaluation:
            #     ssim_mean = np.mean(ssim_list)
            #     psnr_mean = np.mean(psnr_list)
            #     logger.write(['Mean', ssim_mean, psnr_mean])
            #     logger.write([' '])

    # if args.evaluation:
    #     logger.save(os.path.join(output_path, 'result.xlsx'))
    print('Test finished. Save all results to disk.')