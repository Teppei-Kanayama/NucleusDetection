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
from Otsu import Otsu
sys.path.append("../main/")
from unet import UNet
sys.path.append("../preprocess")
from imagetype_classification import imagetype_classification

from morphology import morphology

SIZE = (640, 640)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m',
                        default='/data/unagi0/kanayama/dataset/nuclei_images/checkpoints/CP50.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                        " (default : 'MODEL.pth')")
    parser.add_argument('--gpu', '-g', action='store_true',
                        help="Use the cuda version of the net",
                        default=False)
    parser.add_argument('--test', '-t',
                        help="path to test data",
                        default="/data/unagi0/kanayama/dataset/nuclei_images/stage1_test/")
    parser.add_argument('--save', '-s',
                        help="path to save directory for output masks",
                        default="/data/unagi0/kanayama/dataset/nuclei_images/answer/")

    args = parser.parse_args()
    print("Using model file : {}".format(args.model))
    #net = UNet(3, 1)
    net_gray = UNet(3, 1)
    net_color = UNet(3, 1)
    net_2 = UNet(3, 1)

    if args.gpu:
        print("Using CUDA version of the net, prepare your GPU !")
        #net.cuda()
        net_gray.cuda()
        net_color.cuda()
        net_2.cuda()
    else:
        #net.cpu()
        net_gray.cpu()
        net_color.cpu()
        net_2.cpu()
        print("Using CPU version of the net, this may be very slow")

    print("Loading model ...")
    #net.load_state_dict(torch.load(args.model))
    net_gray.load_state_dict(torch.load('/data/unagi0/kanayama/dataset/nuclei_images/checkpoints/gray_CP100.pth'))
    net_color.load_state_dict(torch.load('/data/unagi0/kanayama/dataset/nuclei_images/checkpoints/color_CP100.pth'))
    net_2.load_state_dict(torch.load('/data/unagi0/kanayama/dataset/nuclei_images/checkpoints/2-CP100.pth'))

    print("Model loaded !")

    for file_name in os.listdir(args.test):
        in_file = args.test + file_name + "/images/" + file_name + ".png"
        out_file = args.save + file_name + ".png"

        print("\nPredicting image {} ...".format(in_file))
        img = Image.open(in_file)
        original_width = img.size[0]
        original_height =  img.size[1]
        img_array = np.asarray(img)

        # 染色方法ごとに処理を分ける
        image_type = imagetype_classification(in_file)
        print(image_type, "\n")
        if image_type == 1:
            out = predict_img(net_gray, img, in_file, args.gpu, SIZE)
            out = morphology(out, iterations=2)
            result = Image.fromarray((out * 255).astype(numpy.uint8))
            result = result.resize((original_width, original_height))
        elif image_type == 2:
            out = predict_img(net_2, img, in_file, args.gpu, SIZE)
            out = morphology(out, iterations=1)
            result = Image.fromarray((out * 255).astype(numpy.uint8))
            result = result.resize((original_width, original_height))
        else:
            out = predict_img(net_color, img, in_file, args.gpu, SIZE)
            out = morphology(out, iterations=2)
            result = Image.fromarray((out * 255).astype(numpy.uint8))
            result = result.resize((original_width, original_height))

        result.save(out_file)
