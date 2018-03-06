import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy
from PIL import Image
import argparse
import os
from utils import *
from unet import UNet
import pdb
import numpy as np


def predict_img(net, img, fn, gpu=False):

    img = np.array(img)
    img = img[:, :, :3] # alphachannelを削除
    img = normalize(img)
    img = np.transpose(img, axes=[2, 0, 1])
    X = torch.FloatTensor(img).unsqueeze(0)

    if gpu:
        X = Variable(X, volatile=True).cuda()
    else:
        X = Variable(X, volatile=True)

    y = F.sigmoid(net(X))
    y = F.upsample_bilinear(y, scale_factor=2).data[0][0].cpu().numpy()

    yy = dense_crf(np.array(img).astype(np.uint8), y)

    return yy > 0.5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='/data/unagi0/kanayama/dataset/nuclei_images/checkpoints/CP50.pth',
                        metavar='FILE',
                        help="Specify the file in which is stored the model"
                        " (default : 'MODEL.pth')")
    parser.add_argument('--cpu', '-c', action='store_true',
                        help="Do not use the cuda version of the net",
                        default=False)
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_false',
                        help="Do not save the output masks",
                        default=False)

    args = parser.parse_args()
    print("Using model file : {}".format(args.model))
    net = UNet(3, 1)
    net_gray = UNet(3, 1)
    net_color = UNet(3, 1)

    if not args.cpu:
        print("Using CUDA version of the net, prepare your GPU !")
        net.cuda()
        net_gray.cuda()
        net_color.cuda()
    else:
        net.cpu()
        net_gray.cpu()
        net_color.cpu()
        print("Using CPU version of the net, this may be very slow")

    TEST_PATH = '/data/unagi0/kanayama/dataset/nuclei_images/stage1_test_preprocessed/images/'
    ANSWER_PATH = '/data/unagi0/kanayama/dataset/nuclei_images/answer/'

    print("Loading model ...")
    net.load_state_dict(torch.load(args.model))
    net_gray.load_state_dict(torch.load('/data/unagi0/kanayama/dataset/nuclei_images/checkpoints/gray_CP50.pth'))
    net_color.load_state_dict(torch.load('/data/unagi0/kanayama/dataset/nuclei_images/checkpoints/color_CP50.pth'))

    print("Model loaded !")

    for file_name in os.listdir(TEST_PATH):
        in_files = [TEST_PATH + file_name]
        out_files = [ANSWER_PATH + file_name]

        original_width =  Image.open("/data/unagi0/kanayama/dataset/nuclei_images/stage1_test/" + file_name.split(".")[0] + "/images/" + file_name).size[0]
        original_height =  Image.open("/data/unagi0/kanayama/dataset/nuclei_images/stage1_test/" + file_name.split(".")[0] + "/images/" + file_name).size[1]

        for i, fn in enumerate(in_files):
            print("\nPredicting image {} ...".format(fn))

            img = Image.open(fn)
            THRESH = 10
            img_array = np.asarray(img)

            if  (img_array[:, :, 1] - img_array[:, :, 2]).sum() ** 2 < THRESH: #grayの場合
                out = predict_img(net_gray, img, fn, not args.cpu)
            else: #colorの場合
                out = predict_img(net_color, img, fn, not args.cpu)

            if args.viz:
                print("Vizualising results for image {}, close to continue ..."
                      .format(fn))

                fig = plt.figure()
                a = fig.add_subplot(1, 2, 1)
                a.set_title('Input image')
                plt.imshow(img)

                b = fig.add_subplot(1, 2, 2)
                b.set_title('Output mask')
                plt.imshow(out)

                plt.show()

            if not args.no_save:
                out_fn = out_files[i]
                result = Image.fromarray((out * 255).astype(numpy.uint8))

                result = result.resize((original_width, original_height))

                result.save(out_files[i])
                print("Mask saved to {}".format(out_files[i]))
