import argparse

from main import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--datasets_name", required=True, type=str, help="name of dataset")
    parser.add_argument("-d", "--datasets_path", required=True, type=str, help="path to folder of dataset")
    parser.add_argument("-p", "--pth_dir", default='./pth', type=str, help="path to save .pth file")
    parser.add_argument("-f", "--fmap", default=64, type=int, help="hyper-parameters for feature map in U-NET")
    parser.add_argument("-g", "--GPU", default="0", type=str, help="gpu(s) to train on")
    parser.add_argument("-e", "--n_epochs", default=30, type=int, help="number of epochs")
    parser.add_argument("-b", "--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("-s", "--save_round", default=10, type=int, help="output cycle of saved model")
    parser.add_argument("-l", "--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("-r", "--gama", default=0.5, type=float, help="hyper-parameters for constraint term in loss function")
    parser.add_argument("-w", "--num_workers", default=4, type=int, help="number of workers to set dataloader")
    parser.add_argument("-t", "--notes", default="train", type=str, help="notes for the experiment")
    parser.add_argument('--validation', action='store_true', help="whether to set up validation during training")
    parser.add_argument("--val_datasize", default=50, type=int, help="max number of slices in the validation output")

    parser.add_argument("--patch_x", default=128, type=int, help="height of each slice in the cropped patch")
    parser.add_argument("--patch_y", default=128, type=int, help="weith of each slice in the cropped patch")
    parser.add_argument("--patch_t", default=32, type=int, help="number of slices in the cropped patch (must be an even number)")
    parser.add_argument("--overlap_factor", default=0.1, type=float, help="factor of overlapping patches")
    parser.add_argument("--norm_mode", default="submean", type=str, help="normalization mode (submean/standard/hard)")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)