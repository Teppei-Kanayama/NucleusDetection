import os
import argparse
from optparse import OptionParser

def parse():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batchsize', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-v', '--val_batch_size', dest='val_batch_size', default=1,
                      type='int', help='validation batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-d', '--data', dest='data',
                      default='/data/unagi0/kanayama/dataset/nuclei_images/stage1_train_default', help='path to training data')
    parser.add_option('-s', '--save', dest='save',
                      default='/data/unagi0/kanayama/dataset/nuclei_images/checkpoints/',
                      help='path to save models')
    parser.add_option('--save_val', dest='save_val',
                      default='/data/unagi0/kanayama/dataset/nuclei_images/answer_val/',
                      help='path to save models')
    parser.add_option('--calc_score', dest='calc_score',
                      default=True,
                      help='whether calculate score or not')

    return parser.parse_args()


def display_info(opt):
    print('+------------------------------+')
    print('| CIFAR classification')
    print('+------------------------------+')
    print('| dataset  : {}'.format(opt.dataset))
    print('| netType  : {}'.format(opt.netType))
    print('| nEpochs  : {}'.format(opt.nEpochs))
    print('| LRInit   : {}'.format(opt.LR))
    print('| schedule : {}'.format(opt.schedule))
    print('| warmup   : {}'.format(opt.warmup))
    print('| batchSize: {}'.format(opt.batchSize))
    print('+------------------------------+')

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(cp), str(gpu)))
