#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : xian
# @Time    : 2019/9/24 21:51

import numpy as np
import torch
import scipy.io as sio
import scipy.optimize.nnls
import scipy
import copy
from VCA import *
from fnnls import *
import pdb
import math


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

class KPmeans(object):
    s_abundance = None
    def __init__(self, argin, verbose):
        self.verbose = verbose
        if (self.verbose):
            print('---		Initializing KPmeans algorithm')
        self.data = argin[0]
        self.nRow = argin[1]
        self.nCol = argin[2]
        self.nBand = argin[3]
        self.nPixel = argin[4]
        self.iter = argin[5]
        self.iter0 = argin[5]
        self.k = argin[6]  # how many endmembers
        #self.snr = argin[7]
        #### add more if need

    def extract_endmember(self):
        self.data = np.asarray(self.data)
        S = np.zeros((self.k,self.nPixel))
        mixed_input = torch.tensor(np.reshape(self.data, [1, self.nBand, self.nRow, self.nCol])).cuda()
        # initA, indice, Yp = vca(self.data, self.k, verbose=True, snr_input=self.snr)
        vca = VCA([self.data, self.nRow, self.nCol, self.nBand, self.nPixel, self.k], self.verbose)
        initA = vca.extract_endmember()[0]
        # initA = sio.loadmat('./DATA/dip_data/unmixing3/a_VCA')['A_vca']
        A_new = torch.tensor(initA).cuda()
        # sio.savemat('./r/data_out_1/' + str(10) + 'initA', {'initA': np.asarray(A_new.cpu())})
        for iter in range(self.iter):
            # E-step
            A_new=A_new.cpu().numpy()
            for ii in range(0,self.nPixel):

                S[:,ii] = fnnls(np.dot(A_new.T, A_new), np.dot(A_new.T, self.data[:,ii]))
            # S = sio.loadmat('./results/data_out_1/SS.mat')['s_abundance']
            #     S[:,ii] = S[:,ii]/sum(S[:,ii])

            # M-step
            # x = torch.zeros(5, 3)
            A_new = torch.tensor(A_new).cuda()
            S1 = torch.tensor(S).cuda()
            values, L = S1.max(0)
            ind = (L, torch.arange(0, self.nPixel))
            L1 = torch.reshape(L, [self.nRow, self.nCol])  # [1,1,512,512]
            print(L.unique())
            if len(L.unique()) != self.k:
                initA = vca.extract_endmember()[0]
                A_new = torch.tensor(initA).cuda()
                self.iter = self.iter0 +iter
                continue
            for ki in range(self.k):
                S2 = torch.tensor(S1)
                S2[ki, :] = 0
                index = (L1 == ki).nonzero()  # identifying positions where the current label is ki
                index_ind = np.ravel_multi_index([index[:, 0].cpu().numpy(), index[:, 1].cpu().numpy()], (self.nRow, self.nCol))
                ind_matched = np.unravel_index(index_ind, (self.nRow, self.nCol))
                Y = (mixed_input - torch.reshape(torch.matmul(A_new, S2), [1, self.nBand, self.nRow, self.nCol])) / torch.reshape(S1[ki, :],[1, 1, self.nRow, self.nCol]).repeat(1, self.nBand, 1, 1)
                # Y = (mixed_input - torch.reshape(torch.matmul(A_new, S2),
                #                                  [1, self.nBand, self.nRow, self.nCol]))
                ak = torch.mean(Y[:, :, ind_matched[0], ind_matched[1]], 2)  # .squeeze(0).squeeze(1)
                A_new[:, ki] = torch.tensor(ak)  # update A

        self.endmember = A_new.cpu().numpy()
        self.abundance = S
        return [self.endmember, self.abundance]