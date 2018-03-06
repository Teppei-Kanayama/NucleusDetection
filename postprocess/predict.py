import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy
from PIL import Image
import argparse
import os
import pdb
import numpy as np

from model import predict_img
import sys
sys.path.append("../main/")
from unet import UNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m',
                        default='/data/unagi0/kanayama/dataset/nuclei_images/checkpoints/CP50.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                        " (default : 'MODEL.pth')")
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--test_preprocessed', '-tp',
                        help="path to preprocessed test data",
                        default="/data/unagi0/kanayama/dataset/nuclei_images/stage1_test_preprocessed/images/")
    parser.add_argument('--test_original', '-to',
                        help="path to preprocessed test data",
                        default="/data/unagi0/kanayama/dataset/nuclei_images/stage1_test/")
    parser.add_argument('--save', '-s',
                        help="path to save directory for output masks",
                        default="/data/unagi0/kanayama/dataset/nuclei_images/answer/")

    args = parser.parse_args()
    print("Using model file : {}".format(args.model))
    net = UNet(3, 1)
    #net_gray = UNet(3, 1)
    #net_color = UNet(3, 1)

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        #net_gray.cuda()
        #net_color.cuda()
    else:
        net.cpu()
        #net_gray.cpu()
        #net_color.cpu()
        print("Using CPU version of the net, this may be very slow")

    print("Loading model ...")
    net.load_state_dict(torch.load(args.model))
    #net_gray.load_state_dict(torch.load('/data/unagi0/kanayama/dataset/nuclei_images/checkpoints/gray_CP50.pth'))
    #net_color.load_state_dict(torch.load('/data/unagi0/kanayama/dataset/nuclei_images/checkpoints/color_CP50.pth'))

    print("Model loaded !")

    for file_name in os.listdir(args.test_preprocessed):
        in_file = args.test_preprocessed + file_name
        out_file = args.save + file_name

        original_img = Image.open(args.test_original + file_name.split(".")[0] + "/images/" + file_name)
        original_width = original_img.size[0]
        original_height =  original_img.size[1]

        print("\nPredicting image {} ...".format(in_file))
        img = Image.open(in_file)
        img_array = np.asarray(img)

        # color画像かモノクロ画像化によって使うモデルを変える場合
        #THRESH = 10
        #if  (img_array[:, :, 1] - img_array[:, :, 2]).sum() ** 2 < THRESH: #grayの場合
        #    out = predict_img(net_gray, img, in_file, not args.cpu)
        #else: #colorの場合
        #    out = predict_img(net_color, img, in_file, not args.cpu)
        out = predict_img(net, img, in_file, not args.cpu)

        result = Image.fromarray((out * 255).astype(numpy.uint8))
        result = result.resize((original_width, original_height))

        result.save(out_file)
        print("Mask saved to {}".format(out_file))
