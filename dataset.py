import os
import random
import math
import torch

import numpy as np
import tifffile as tiff

from torch.utils.data import Dataset


def normalize(x, mode='submean'):
    if mode == 'submean':
        m = x.mean()
        x = x - m
        return x, m
    elif mode == 'standard':
        m = x.mean()
        s = x.std()
        x = (x - m) / (s + 1e-6)
        return x, (m, s)
    elif mode == 'hard':
        p1 = x.min()
        p2 = x.max()
        x = (x - p1) / (p2 - p1)
        return x, (p1, p2)
    else:
        raise ValueError(f'No such normlize mode: {mode}')
    

def inv_normalize(x, param, mode='submean'):
    if mode == 'submean':
        x = x + param
    elif mode == 'standard':
        x = x * param[1] + param[0]
    elif mode == 'hard':
        x = x * (param[1] - param[0]) + param[0]
    else:
        raise ValueError(f'No such normlize mode: {mode}')
    return x


def random_transform(input):
    """
    The function for data augmentation. Randomly select one method among five
    transformation methods (including rotation and flip) or do not use data
    augmentation.

    Args:
        input, target : the input and target patch before data augmentation
    Return:
        input, target : the input and target patch after data augmentation
    """
    p_flip = random.randrange(4)
    p_rot = random.randrange(4)
    if p_flip == 0:
        input = input
    elif p_flip == 1:
        input = input[:, :, ::-1]
    elif p_flip == 2:
        input = input[::-1, :, :]
    elif p_flip == 3:
        input = input[::-1, :, :]
        input = input[:, :, ::-1]
        
    if p_rot == 0:  # no transformation
        input = input
    elif p_rot == 1:  # left rotate 90
        input = np.rot90(input, k=1, axes=(1, 2))
    elif p_rot == 2:  # left rotate 180
        input = np.rot90(input, k=2, axes=(1, 2))
    elif p_rot == 3:  # left rotate 270
        input = np.rot90(input, k=3, axes=(1, 2))

    return input


class trainset(Dataset):
    """
    Train set generator for pytorch training

    """

    def __init__(self, name_list, coordinate_list, noise_img_all, stack_index):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.noise_img_all = noise_img_all
        self.stack_index = stack_index

    def __getitem__(self, index):
        """
        For temporal stacks with a small lateral size or short recording period, sub-stacks can be
        randomly cropped from the original stack to augment the training set according to the record
        coordinate. Then, interlaced frames of each sub-stack are extracted to form two 3D tiles.
        One of them serves as the input and the other serves as the target for network training
        Args:
            index : the index of 3D patchs used for training
        Return:
            input, target : the consecutive frames of the 3D noisy patch serve as the input and target of the network
        """
        stack_index = self.stack_index[index]
        noise_img = self.noise_img_all[stack_index]
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']
        patch = noise_img[init_s:end_s, init_h:end_h, init_w:end_w]
        patch = random_transform(patch)

        patch = torch.from_numpy(np.expand_dims(patch, 0).copy())
        return patch

    def __len__(self):
        return len(self.name_list)


class testset(Dataset):
    """
    Test set generator for pytorch inference

    """

    def __init__(self, name_list, coordinate_list, noise_img):
        self.name_list = name_list
        self.coordinate_list = coordinate_list
        self.noise_img = noise_img

    def __getitem__(self, index):
        """
        Generate the sub-stacks of the noisy image.
        Args:
            index : the index of 3D patch used for testing
        Return:
            noise_patch : the sub-stacks of the noisy image
            single_coordinate : the specific coordinate of sub-stacks in the noisy image for stitching all sub-stacks
        """
        single_coordinate = self.coordinate_list[self.name_list[index]]
        init_h = single_coordinate['init_h']
        end_h = single_coordinate['end_h']
        init_w = single_coordinate['init_w']
        end_w = single_coordinate['end_w']
        init_s = single_coordinate['init_s']
        end_s = single_coordinate['end_s']
        noise_patch = self.noise_img[init_s:end_s, init_h:end_h, init_w:end_w]
        noise_patch = torch.from_numpy(np.expand_dims(noise_patch, 0))
        return noise_patch, single_coordinate

    def __len__(self):
        return len(self.name_list)


def train_preprocess(args):
    """
    The original noisy stack is partitioned into thousands of 3D sub-stacks (patch) with the setting
    overlap factor in each dimension.

    Important Fields:
        self.name_list : the coordinates of 3D patch are indexed by the patch name in name_list.
        self.coordinate_list : record the coordinate of 3D patch preparing for partition in whole stack.
        self.stack_index : the index of the noisy stacks.
        self.noise_im_all : the collection of all noisy stacks.

    """
    name_list = []
    coordinate_list = {}
    stack_index = []
    noise_im_all = []
    ind = 0

    gap_x = int(args.patch_x * (1 - args.overlap_factor))  # patch gap in x
    gap_y = int(args.patch_y * (1 - args.overlap_factor))  # patch gap in y
    gap_t = int(args.patch_t * (1 - args.overlap_factor))  # patch gap in t
    
    def round_diy(x, w, p, g):
        b = int(x)
        t = math.ceil(x)
        
        if (b >= (w-2*p)/g+2) and (b > 1):
            return b
        else:
            return t

    print('\033[1;31m===== Stack Information ===== \033[0m')
    for i, im_name in enumerate(list(os.walk(args.datasets_path, topdown=False))[-1][-1]):
        im_dir = args.datasets_path + '//' + im_name
        noise_im = tiff.imread(im_dir)
        whole_x = noise_im.shape[2]
        whole_y = noise_im.shape[1]
        whole_t = noise_im.shape[0]
        print(f'[{i+1}] {im_name} {noise_im.shape}')

        noise_im = noise_im.astype(np.float32)
        noise_im, _ = normalize(noise_im, mode=args.norm_mode)

        noise_im_all.append(noise_im)
        h_num = round_diy((whole_y - args.patch_y + gap_y) / gap_y, whole_y, args.patch_y, gap_y)
        w_num = round_diy((whole_x - args.patch_x + gap_x) / gap_x, whole_x, args.patch_x, gap_x)
        s_num = round_diy((whole_t - args.patch_t + gap_t) / gap_t, whole_t, args.patch_t, gap_t)
        for x in range(0, h_num):
            for y in range(0, w_num):
                for z in range(0, s_num):
                    single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}

                    if x < h_num - 1:
                        init_h = gap_y * x
                        end_h = gap_y * x + args.patch_y
                    else:
                        end_h = whole_y
                        init_h = whole_y - args.patch_y
                    if y < w_num - 1:
                        init_w = gap_x * y
                        end_w = gap_x * y + args.patch_x
                    else:
                        end_w = whole_x
                        init_w = whole_x - args.patch_x
                    if z < s_num - 1:
                        init_s = gap_t * z
                        end_s = gap_t * z + args.patch_t
                    else:
                        end_s = whole_t
                        init_s = whole_t - args.patch_t

                    single_coordinate['init_h'] = init_h
                    single_coordinate['end_h'] = end_h
                    single_coordinate['init_w'] = init_w
                    single_coordinate['end_w'] = end_w
                    single_coordinate['init_s'] = init_s
                    single_coordinate['end_s'] = end_s

                    patch_name = args.datasets_name + '_' + im_name.replace('.tif', '') + '_x' + str(
                        x) + '_y' + str(y) + '_z' + str(z)
                    name_list.append(patch_name)
                    coordinate_list[patch_name] = single_coordinate
                    stack_index.append(ind)
        ind = ind + 1
    print()
    
    return name_list, coordinate_list, stack_index, noise_im_all


def test_preprocess(args, img_id):
    """
    Choose one original noisy stack and partition it into thousands of 3D sub-stacks (patch) with the setting
    overlap factor in each dimension.

    Args:
        args : the train object containing input params for partition
        img_id : the id of the test image
    Returns:
        name_list : the coordinates of 3D patch are indexed by the patch name in name_list
        noise_im : the original noisy stacks
        coordinate_list : record the coordinate of 3D patch preparing for partition in whole stack
        im_name : the file name of the noisy stacks

    """
    patch_y = args.patch_y
    patch_x = args.patch_x
    patch_t = args.patch_t
    gap_x = int(args.patch_x * (1 - args.overlap_factor))  # patch gap in x
    gap_y = int(args.patch_y * (1 - args.overlap_factor))  # patch gap in y
    gap_t = int(args.patch_t * (1 - args.overlap_factor))  # patch gap in t
    
    cut_w = (patch_x - gap_x) / 2
    cut_h = (patch_y - gap_y) / 2
    cut_s = (patch_t - gap_t) / 2
    im_folder = args.datasets_path

    name_list = []
    coordinate_list = {}
    img_list = list(os.walk(im_folder, topdown=False))[-1][-1]
    img_list.sort()

    im_name = img_list[img_id]

    im_dir = im_folder + '//' + im_name
    noise_im = tiff.imread(im_dir)
    input_data_type = noise_im.dtype

    print(f'Stack information: {im_name} {noise_im.shape}')
    noise_im = noise_im.astype(np.float32)
    noise_im, norm_param = normalize(noise_im, args.norm_mode)

    whole_x = noise_im.shape[2]
    whole_y = noise_im.shape[1]
    whole_t = noise_im.shape[0]

    num_w = math.ceil((whole_x - patch_x + gap_x) / gap_x)
    num_h = math.ceil((whole_y - patch_y + gap_y) / gap_y)
    num_s = math.ceil((whole_t - patch_t + gap_t) / gap_t)

    for x in range(0, num_h):
        for y in range(0, num_w):
            for z in range(0, num_s):
                single_coordinate = {'init_h': 0, 'end_h': 0, 'init_w': 0, 'end_w': 0, 'init_s': 0, 'end_s': 0}
                if x != (num_h - 1):
                    init_h = gap_y * x
                    end_h = gap_y * x + patch_y
                elif x == (num_h - 1):
                    init_h = whole_y - patch_y
                    end_h = whole_y

                if y != (num_w - 1):
                    init_w = gap_x * y
                    end_w = gap_x * y + patch_x
                elif y == (num_w - 1):
                    init_w = whole_x - patch_x
                    end_w = whole_x

                if z != (num_s - 1):
                    init_s = gap_t * z
                    end_s = gap_t * z + patch_t
                elif z == (num_s - 1):
                    init_s = whole_t - patch_t
                    end_s = whole_t
                single_coordinate['init_h'] = init_h
                single_coordinate['end_h'] = end_h
                single_coordinate['init_w'] = init_w
                single_coordinate['end_w'] = end_w
                single_coordinate['init_s'] = init_s
                single_coordinate['end_s'] = end_s

                if y == 0:
                    single_coordinate['stack_start_w'] = y * gap_x
                    single_coordinate['stack_end_w'] = y * gap_x + patch_x - cut_w
                    single_coordinate['patch_start_w'] = 0
                    single_coordinate['patch_end_w'] = patch_x - cut_w
                elif y == num_w - 1:
                    single_coordinate['stack_start_w'] = whole_x - patch_x + cut_w
                    single_coordinate['stack_end_w'] = whole_x
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x
                else:
                    single_coordinate['stack_start_w'] = y * gap_x + cut_w
                    single_coordinate['stack_end_w'] = y * gap_x + patch_x - cut_w
                    single_coordinate['patch_start_w'] = cut_w
                    single_coordinate['patch_end_w'] = patch_x - cut_w

                if x == 0:
                    single_coordinate['stack_start_h'] = x * gap_y
                    single_coordinate['stack_end_h'] = x * gap_y + patch_y - cut_h
                    single_coordinate['patch_start_h'] = 0
                    single_coordinate['patch_end_h'] = patch_y - cut_h
                elif x == num_h - 1:
                    single_coordinate['stack_start_h'] = whole_y - patch_y + cut_h
                    single_coordinate['stack_end_h'] = whole_y
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y
                else:
                    single_coordinate['stack_start_h'] = x * gap_y + cut_h
                    single_coordinate['stack_end_h'] = x * gap_y + patch_y - cut_h
                    single_coordinate['patch_start_h'] = cut_h
                    single_coordinate['patch_end_h'] = patch_y - cut_h

                if z == 0:
                    single_coordinate['stack_start_s'] = z * gap_t
                    single_coordinate['stack_end_s'] = z * gap_t + patch_t - cut_s
                    single_coordinate['patch_start_s'] = 0
                    single_coordinate['patch_end_s'] = patch_t - cut_s
                elif z == num_s - 1:
                    single_coordinate['stack_start_s'] = whole_t - patch_t + cut_s
                    single_coordinate['stack_end_s'] = whole_t
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t
                else:
                    single_coordinate['stack_start_s'] = z * gap_t + cut_s
                    single_coordinate['stack_end_s'] = z * gap_t + patch_t - cut_s
                    single_coordinate['patch_start_s'] = cut_s
                    single_coordinate['patch_end_s'] = patch_t - cut_s

                patch_name = args.datasets_name + '_x' + str(x) + '_y' + str(y) + '_z' + str(z)
                name_list.append(patch_name)
                coordinate_list[patch_name] = single_coordinate

    return name_list, noise_im, coordinate_list, im_name, norm_param, input_data_type