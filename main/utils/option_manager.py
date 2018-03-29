import os
from optparse import OptionParser

def parse():
    parser = OptionParser()
    parser.add_option('--id', type='int', default=0,
                      help='ID of Experiment')
    parser.add_option('--epochs', default=300, type='int',
                      help='number of epochs')
    parser.add_option('--batchsize', default=12,
                      type='int', help='batch size')
    parser.add_option('--val_batchsize', default=1,
                      type='int', help='validation batch size')
    parser.add_option('--val_percent', default=0.05,
                      type='float', help='validation batch size')
    parser.add_option('--lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('--gpu', action='store_true',
                      default=False, help='use cuda')
    parser.add_option('--load_model',
                      default=False, help='load file model')
    parser.add_option('--load_state',
                      default=False, help='load file state')
    parser.add_option('--data',
                      default='/data/unagi0/kanayama/dataset/nuclei_images/stage1_train_default', help='path to training data')
    parser.add_option('--save_model',
                      default='/data/unagi0/kanayama/dataset/nuclei_images/checkpoints/',
                      help='path to save models')
    parser.add_option('--save_state',
                      default='/data/unagi0/kanayama/dataset/nuclei_images/checkpoints/',
                      help='path to save states')
    parser.add_option('--save_val',
                      default='/data/unagi0/kanayama/dataset/nuclei_images/answer_val',
                      help='path to save models')
    parser.add_option('--skip_calc_score', action='store_true',
                      default=False, help='whether calculate score or not')
    parser.add_option('--skip_train', action='store_true', default=False,
                      help='skip training phase')
    parser.add_option('--save_probs', action='store_true', default=False,
                      help="save the probability of model's output")
    parser.add_option('--resize_shape', default=(256, 256),
                      help='the shape to resize image')
    parser.add_option('--calc_score_step', default=5,
                      help='interval to calculate score')


    return parser.parse_args()


def display_info(options, N_train, N_val):
    print('+------------------------------+')
    print('| U-Net Segmentation')
    print('+------------------------------+')
    print('| Epochs  : {}'.format(options.epochs))
    print('| Batch size  : {}'.format(options.batchsize))
    print('| Learning rate  : {}'.format(options.lr))
    print('| Training size   : {}'.format(N_train))
    print('| Validation size : {}'.format(N_val))
    print('| Cuda   : {}'.format(options.gpu))
    print('| Resize Shape   : {}'.format(options.resize_shape))
    print('+------------------------------+')
