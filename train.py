import os
import torch
import argparse
import random
import numpy as np
from IQASolver import demoIQASolver
import torchvision


torch.cuda.set_device(0)


def main(config):
    if config.mode == "train":
        folder_path = {
            'live': '../DATE/live/',
            'csiq': '../DATE/csiq',
            'livec': '../DATE/livec/',
            'livemd':'../DATE/livemd'
        }
        img_num = {
            'live': list(range(0, 29)),
            'csiq': list(range(0, 30)),
            'livec': list(range(0, 1162)),
            'livemd': list(range(0, 450)),
        }
        sel_num = img_num[config.dataset]
        srcc_all = np.zeros(config.train_test_num, dtype=np.float32)
        plcc_all = np.zeros(config.train_test_num, dtype=np.float32)
        print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
        for i in range(config.train_test_num):
            print('Round %d' % (i + 1))
            # Randomly select 80% images for training and the rest for testing
            random.shuffle(sel_num)
            train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
            test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
            solver = demoIQASolver(config, folder_path[config.dataset], train_index, test_index)
            srcc_all[i], plcc_all[i] = solver.train()
        srcc_med = np.mean(srcc_all)
        plcc_med = np.mean(plcc_all)
        print('Testing mean SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))
if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', type=str, default='train')
    parser.add_argument('--dataset', dest='dataset', type=str, default=' ', help='Support datasets: live|csiq|livec|livemd')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default= )
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default= )
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default= )
    parser.add_argument('--batch_size', dest='batch_size', type=int, default= )
    parser.add_argument('--epochs', dest='epochs', type=int, default= )
    parser.add_argument('--patch_size', dest='patch_size', type=int, default= )
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default= )

    config = parser.parse_args()
    main(config)
