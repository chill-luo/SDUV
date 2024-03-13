import argparse

from main import test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--datasets_name", required=True, type=str, help="name of dataset")
    parser.add_argument("-d", "--datasets_path", required=True, type=str, help="path to folder of dataset")
    parser.add_argument("-o", "--output_dir", default='./results', type=str, help="path to save output files")
    parser.add_argument("-p", "--pth_dir", default='./pth', type=str, help="path to load .pth file(s)")
    parser.add_argument("-m", "--denoise_model", required=True, type=str, help="name of the experimental folder to be loaded")
    parser.add_argument("-s", "--pths_num", default="20,30", type=str, help="serial number of the loaded. pth files")
    parser.add_argument("-f", "--fmap", default=64, type=int, help="hyper-parameters for feature map in U-NET")
    parser.add_argument("-g", "--GPU", default="0", type=str, help="gpu(s) to train on")
    parser.add_argument("-w", "--num_workers", default=4, type=int, help="number of workers to set dataloader")

    parser.add_argument("--patch_x", default=128, type=int, help="height of each slice in the cropped patch")
    parser.add_argument("--patch_y", default=128, type=int, help="weith of each slice in the cropped patch")
    parser.add_argument("--patch_t", default=16, type=int, help="number of slices in the cropped patch")
    parser.add_argument("--overlap_factor", default=0.1, type=float, help="factor of overlapping patches")
    parser.add_argument("--norm_mode", default="submean", type=str, help="normalization mode (submean/standard/hard)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    test(args)