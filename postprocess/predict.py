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
from skimage import transform

from model import predict_img
import sys
from Otsu import Otsu
from Devide import Devide
sys.path.append("../main/")
from unet import UNet
sys.path.append("../preprocess")
from imagetype_classification import imagetype_classification

from morphology import morphology
from resize import resize
from remove_noise import remove_noise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m',
                        default='/data/unagi0/kanayama/dataset/nuclei_images/checkpoints/5_CP300.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model")
    parser.add_argument('--gpu', '-g', action='store_true',
                        help="Use the cuda version of the net",
                        default=False)
    parser.add_argument('--test', '-t',
                        help="path to test data",
                        default="/data/unagi0/kanayama/dataset/nuclei_images/stage1_test/")
    parser.add_argument('--save', '-s',
                        help="path to save directory for output masks",
                        default="/data/unagi0/kanayama/dataset/nuclei_images/answer/")
    parser.add_argument('--size',
                        help="the size when input U-Net", default=640)

    args = parser.parse_args()
    print("Using model file : {}".format(args.model))
    net = UNet(3, 1)

    if args.gpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
    else:
        print("Using CPU version of the net, this may be very slow")
        net.cpu()

    print("Loading model ...")
    net.load_state_dict(torch.load(args.model))

    print("Model loaded !")

    for file_name in os.listdir(args.test):
        in_file = args.test + file_name + "/images/" + file_name + ".png"
        out_file = args.save + file_name + ".png"

        print("\nPredicting image {} ...".format(in_file))
        original_img = Image.open(in_file)

        # 元画像の大きさを保存しておく
        original_width = original_img.size[0]
        original_height =  original_img.size[1]
        original_img = np.asarray(original_img)

        # 染色方法を識別
        # image_type = imagetype_classification(in_file)

        # 所定の大きさにresize
        resized_img = resize(original_img, args.size)

        # U-Netを用いてsegentation
        dst_img = predict_img(net, resized_img, args.gpu)

        # 後処理
        #dst_img_array = morphology(dst_img_array, iterations=1)

        # もとの大きさに戻す
        dst_img_resized = (dst_img * 255).astype(np.uint8)

        # 結果を画像として保存
        result = Image.fromarray(dst_img_resized)
        result.save(out_file)
