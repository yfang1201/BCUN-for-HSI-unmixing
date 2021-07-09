#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : xian
# @Time    : 2019/11/15 15:06

from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib
import scipy.io as sio
import matplotlib
import statistics
import numpy.matlib
from torchsummary import summary

import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from scipy import signal

import numpy as np
from models import *
# from VCA2 import *
from common_utils import *
import scipy.misc
from PIL import Image
from PIL import ImageFilter
import pandas as pd

import torch
import torch.optim
import matplotlib.pyplot as plt
from torchviz import make_dot

import pdb
from sklearn.metrics import mean_squared_error
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
from models.downsampler import Downsampler
from VCA import *

from utils.sr_utils import *
import warnings
warnings.simplefilter('ignore')

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

PLOT = True


class AEsmm_mse(object):
    data = None
    nRow = None
    nCol = None
    nBand = None
    nPixel = None
    p = None
    net = None
    endmember = None
    abundance = None

    verbose = True


    def __init__(self, argin, verbose):
        self.verbose = verbose
        if (self.verbose):
            print('---		Initializing AESMM MSE algorithm')
        self.data = torch.tensor(argin[0]).type(dtype) # url
        self.nRow = argin[1]  #
        self.nCol = argin[2]  #
        self.nBand = argin[3]  #
        self.npixel = argin[4]
        self.k = argin[5]  # how many endmembers
        self.iterEM = argin[6]  # the total interation times ?????????there is an "iteration" outside
        self.iterEM0 = argin[6]
        self.iter = argin[7]  # the iteration to train the encoder network
        self.INPUT = argin[8]  # 'noise'
        self.pad = argin[9]  # 'reflection'
        self.OPT_OVER = argin[10]  # 'net'
        self.LR = argin[11]  # 0.0005  #0.0004 for SNR20
        self.tv_weight = argin[12]  # 0.0
        self.OPTIMIZER = argin[13]  # 'adam'
        self.NET_TYPE = argin[14]  # 'skip'
        self.net_input = torch.reshape(torch.tensor(argin[15]).cuda().type(dtype), [1, self.nBand, self.nRow, self.nCol])
        self.var_noi = torch.tensor(argin[16]).cuda().type(dtype)
        self.A_true = argin[17]
        self.S_true = argin[18]

        self.upsample_mode = 'bilinear'
        self.skip_n33d = 128
        self.skip_n33u = 128
        self.skip_n11 = 4
        self.num_scales = 3
        self.act_fun = 'LeakyReLU'
        self.downsample_mode = 'stride'


    def extract_endmember(self):
        '''mse_history = []
        ssim_history = []
        loss_history = []'''

        net = get_net(self.nBand, self.NET_TYPE, self.pad, self.upsample_mode, self.k, self.act_fun, self.skip_n33d, self.skip_n33u, self.skip_n11, self.num_scales, self.downsample_mode).type(dtype)
        net = net.float().cuda()
        # mse = torch.nn.MSELoss().type(dtype)

        #loc = np.random.randint(self.nRow * self.nCol + 1, size=4)
        #initA = self.data[:, loc].squeeze()
        #pdb.set_trace()
        vca = VCA([self.data.cpu().numpy(), self.nRow, self.nCol, self.nBand, self.npixel, self.k], self.verbose)
        initA = vca.extract_endmember()[0]
        # initA, indice, Yp = vca(self.data.cpu().numpy(), self.k, verbose=True, snr_input=0)

        A_new = torch.tensor(initA).cuda()
        #pdb.set_trace()
        mixed_input = torch.reshape(self.data, [1, self.nBand, self.nRow, self.nCol]).type(dtype)

        def closure():
            #global i
            #pdb.set_trace()
            S = net(self.net_input)  # encoder   S: 1 * k * self.nRow * self.nCol
            out_mixed = torch.reshape(
                torch.matmul(A_new.detach().float(), torch.reshape(S, (1 * self.k, self.npixel))),
                (1, self.nBand, self.nRow, self.nCol))  # decoder A:k*P
            #total_loss = mse(out_mixed, mixed_input)
            total_loss = EdisLoss(out_mixed, mixed_input, self.var_noi)
            if self.tv_weight > 0:
                total_loss += float(self.tv_weight * tv_loss(S))
            total_loss.backward()
            #i += 1
            print(total_loss.item())
            return total_loss

        for iter_em in range(self.iterEM):
            print('EM iter :', iter_em)
            # i = 0
            para = get_params(self.OPT_OVER, net,
                                self.data)  # the 4th argument should be net_input. Using mixed_input does not affacts the results in this code.

            optimize(self.OPTIMIZER, para, closure, self.LR, self.iter)

            '''with torch.no_grad():
                S = net(self.net_input)  # size: 1 * k * rows * columns
                S1 = torch.reshape(S, [self.k, self.npixel])
                #print(S1)
                for ki in range(self.k):
                    S0 = torch.tensor(S1)
                    S0[ki, :] = 0
                    Y = (mixed_input - torch.reshape(torch.matmul(A_new, S0), [1, self.nBand, self.nRow, self.nCol])) \
                        / torch.reshape(S1[ki, :], [1, 1, self.nRow, self.nCol]).repeat(1, self.nBand, 1, 1)
                    #pdb.set_trace()
                    ak = torch.sum(S1[ki, :].repeat([self.nBand, 1]) * torch.reshape(Y, [self.nBand, self.npixel]), 1) \
                         / torch.sum(S1[ki, :]) #/ (self.var_noi-torch.mean(self.var_noi,1)+1)
                    # k = torch.median(torch.reshape(Y,[channels,rows*columns]), dim=1).values
                    A_new[:, ki] = torch.tensor(ak)  # update A'''

            # S_np = torch_to_np(S)
            # S_img = np.reshape(S_np[0, :, :], (1, self.nRow, self.nCol))
            # S_true = self.S_true.reshape(self.k, self.nRow, self.nCol).transpose([0, 2, 1]).reshape(self.k, self.npixel)  # k*(rows*columns)
            # S_true_img = np.reshape(S_true, (self.k, self.nRow, self.nCol))
            # plot_image_grid([S_img, np.expand_dims(S_true_img[0, :, :], 0)], factor=8, nrow=2)
            # plt.plot(A_new[:, 1].cpu().numpy().squeeze(), color='red', linewidth=1)
            # plt.plot(self.A_true[:, 1], color='orange', linewidth=1.5)
            # plt.show()


            with torch.no_grad():
                S = net(self.net_input)
                S1 = torch.reshape(S, [self.k, self.npixel])
                values, L = S1.max(0)
                ind = (L, torch.arange(0, self.npixel))
                L1 = torch.reshape(L, [self.nRow, self.nCol])  # [1,1,512,512]
                print(L.unique())
                if len(L.unique()) != self.k:
                    initA = vca.extract_endmember()[0]
                    A_new = torch.tensor(initA).cuda()
                    self.iterEM = self.iterEM0+iter_em
                    continue

                for ki in range(self.k):
                    #sort, indices = torch.sort(S1[ki, :], descending=True)  # indices:10816*1
                    S2 = torch.tensor(S1)
                    #pdb.set_trace()
                    S2[ki, :] = 0
                    index = (L1 == ki).nonzero()
                    # identifying positions where the current label is ki
                    #index_ind = (S1[ki, :] > 0.1).nonzero().cpu().numpy()
                    index_ind = np.ravel_multi_index([index[:, 0].cpu().numpy(), index[:, 1].cpu().numpy()], (self.nRow, self.nCol))
                    ind_matched = np.unravel_index(index_ind, (self.nRow, self.nCol))
                    Y = (mixed_input - torch.reshape(torch.matmul(A_new, S2), [1, self.nBand, self.nRow, self.nCol]))/torch.reshape(S1[ki, :], [1, 1, self.nRow, self.nCol]).repeat(1, self.nBand, 1, 1)
                    ak = torch.mean(Y[:, :, ind_matched[0], ind_matched[1]], 2)#.squeeze(0).squeeze(1)
                    A_new[:, ki] = torch.tensor(ak)  # update A
                    #pdb.set_trace()


        self.endmember = A_new.cpu().numpy()
        self.abundance = np.reshape(S.cpu().numpy(), [self.k, self.npixel])
        return [self.endmember, self.abundance]
