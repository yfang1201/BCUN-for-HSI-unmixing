#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : xian
# @Time    : 2019/10/23 19:42

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : xian
# @Time    : 2019/8/21 10:33

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import scipy.io as sio
import scipy.optimize.nnls
import scipy
import numpy as np
from VCA import *
from PPI import *
from NFINDR import *
from MVCNMF import *
from AEsmm import *
import matplotlib.pyplot as plt
import torch
import torchvision
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_mse as mse
from sklearn.utils import shuffle
from tabulate import tabulate
import pandas as pd
from timeit import default_timer as timer
from common_utils import *
from tempfile import TemporaryFile
import pdb

class DEMO(object):
    name = None
    ee = None  # 提取端元方法
    data = None
    nRow = None
    nCol = None
    nBand = None
    nPixel = None
    p = None
    groundtruth = None
    abf_gt = None
    num_gtendm = None
    raw_endmembers = None
    abundances = None
    S = None
    sad_em_max = None
    sad_em_min = None
    aad_em_max = None
    aad_em_min = None
    sad_ab_max = None
    sad_ab_min = None
    aad_ab_max = None
    aad_ab_min = None
    sad_values_max = None
    sad_values_min = None
    aad_values_max = None
    aad_values_min = None
    sad_max = None
    sad_min = None
    aad_max = None
    aad_min = None
    sad_mean = None
    sad_var = None
    sad_std = None
    aad_mean = None
    aad_var = None
    aad_std = None
    sid_em_max = None
    sid_em_min = None
    aid_em_max = None
    aid_em_min = None
    sid_values_max = None
    sid_values_min = None
    aid_values_max = None
    aid_values_min = None
    sid_max = None
    sid_min = None
    aid_max = None
    aid_min = None
    sid_mean = None
    sid_var = None
    sid_std = None
    aid_mean = None
    aid_var = None
    aid_std = None
    sad_all_runs_value = None
    sid_all_runs_value = None
    aad_all_runs_value = None
    aid_all_runs_value = None
    time_runs = None
    verbose = True


    def __init__(self, argin, verbose):
        if (verbose):
            print('... Initializing DEMO')
        self.verbose = verbose    # 是否在屏幕打印
        # self.load_data(argin[0])   # 加载混合影像数据
        # self.data = np.asmatrix(self.addnoise(self.data_clean, g)[0])
        self.load_groundtruth(argin[1])    # 加载真实标签数据
        self.load_abf_true(abf_true_loc)    # 加载真实丰度值
        # self.data_clean = self.convert_2D(self.data_clean)    # 三维影像转为二维数据
        self.p = argin[2]    # 端元数
        self.name = argin[3]    # 端元提取的方法
        self.nRow = 104
        self.nCol = 104
        self.nPixel = self.nRow * self.nCol
        global nR, nC
        nR = self.nRow
        nC = self.nCol

    def NNLS(self, Y, A, iteraNum, firstUpdateMatrixFlag):
        if firstUpdateMatrixFlag == 1:
            # ====================================== #
            # initialize modelOrder and A , S matrix #
            # ====================================== #
            modelOrder = A.shape[1]
            # A takes the input matrix as initial matrix
            # or reuse the updated matrix in the last iteration
            S = np.zeros([modelOrder, Y.shape[1]])
            itera = 0
            for itera in range(0, iteraNum):    # 丰度估计的迭代次数
                print ("iteration : %d/%d" % (itera + 1, iteraNum))
                for j in range(0, Y.shape[1]):    # 逐像素
                    # ----------------------------- #
                    # prepare a Y column vector Y_j #
                    # ----------------------------- #
                    Y_j = np.zeros([1, Y.shape[0]])[0]
                    #             conversion to array for scipy requires
                    #             a row matrix such that the shape is
                    #             taken as a transpose
                    #             Y_j = np.zeros( [ Y.shape[0] , 1 ] )
                    for k in range(0, Y.shape[0]):    # 逐波段
                        Y_j[k] = Y[k, j]    # 获取每个像素在220个波段上的值
                    # --------- #
                    # NNLS part #
                    # --------- #
                    S_j = scipy.optimize.nnls(A, Y_j)[0]    # A是提取的端元值，Y_j是每个像素在220个波段上的光谱值--->S_j：求的的该像素种四种端元的丰度值
                    # ----------------------------------- #
                    # update the obatined optimized       #
                    # value of the jth column to S matrix #
                    # ----------------------------------- #
                    for k in range(0, S_j.size):
                        S[k, j] = S_j[k]
                # ~~~~~~~~~~~~~ #
                # Update A Part #
                # ~~~~~~~~~~~~~ #
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
                # mathematical model :          #
                # Yt = St * At                  #
                # where Xt means transpose of X #
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
                Yt = Y.transpose()    # 影像数据Y的转置
                At = A.transpose()    # 端元值的转置
                St = S.transpose()    # 丰度值的转置
                # --------------------------------------- #
                # in min( || Yt - St * At ||2 )           #
                #    sub to At geq 0                      #
                # for each column of At , perform NNLS    #
                # by using St matrix and ith column of Yt #
                # ith column is indicated by Yt_j         #
                # --------------------------------------- #
                for j in range(0, Yt.shape[1]):    # 逐波段
                    # ------------------------------- #
                    # prepare a Yt column vector Yt_j #
                    # ------------------------------- #
                    Yt_j = np.zeros([1, Yt.shape[0]])[0]
                    #             conversion to array for scipy requires
                    #             a row matrix such that the shape is
                    #             taken as a transpose
                    #             Yt_j = np.zeros( [ Yt.shape[0] , 1 ] )
                    for k in range(0, Yt.shape[0]):  # 逐像素
                        Yt_j[k] = Yt[k, j]
                    # --------- #
                    # NNLS part #
                    # --------- #
                    At_j = scipy.optimize.nnls(St, Yt_j)[0]    # 再根据求得的丰度值反求端元值，进行优化  At_j.size:(4,)
                    # ------------------------------------ #
                    # update the obtained optimized        #
                    # value of the jth column to At matrix #
                    # ------------------------------------ #
                    for k in range(0, At_j.size):
                        At[k, j] = At_j[k]    # 最后求得4种端元的新端元值：4*220
                # --------------------------------------------- #
                # update A after the above transposed NNLS      #
                # no need to update S matrix as it is unchanged #
                # --------------------------------------------- #
                A = At.transpose()
            # ===================== #
            # End of NNLS iteration #
            # ===================== #
            S = np.reshape(S, [self.num_gtendm, self.nRow, self.nCol])
            S = S.transpose((0, 2, 1))
            S = np.reshape(S, [self.num_gtendm, self.nPixel])
            #pdb.set_trace()
            return A, S

    def load_data(self, data_loc):
        if (verbose):
            print('... Loading data')

        pkg_data = sio.loadmat(data_loc)
        self.data_clean = pkg_data['mixed'][:,:,0:200]

        self.nRow = self.data_clean.shape[0]
        self.nCol = self.data_clean.shape[1]
        self.nBand = self.data_clean.shape[2]

        global nR, nC
        nR = self.nRow
        nC = self.nCol

    def addnoise(self, img, SNR):
        indian_pine_loc = "./DATA/Indian_pines_corrected.mat"
        indian_pine = sio.loadmat(indian_pine_loc)['indian_pines_corrected']
        ROI_indian_pine = np.reshape(indian_pine[100:113, 27:45, :], (234, 200))

        n = ROI_indian_pine.shape[0]

        var_noi = np.var(ROI_indian_pine, axis=0, keepdims=True).squeeze()
        tmp = np.sum(pow(ROI_indian_pine, 2), 0) / (var_noi * n)
        snr_noi = 10 * np.log10(tmp)
        snr_noi[snr_noi < 0] = 0
        cen_snr_noi = (snr_noi - np.mean(snr_noi))  ## this is the centered snr
        nor_cen_snr_noi = cen_snr_noi / np.max(cen_snr_noi)  ## this is the centered and normalized snr
        scal_nor_cen_snr_noi = 5 * nor_cen_snr_noi

        # add noise to the simulate data
        mixedpure = np.reshape(img, [self.nRow * self.nCol, self.nBand])

        snr = SNR + scal_nor_cen_snr_noi
        var_img = np.sum((np.power(mixedpure, 2)), 0) / np.power(10, snr / 10) / self.nRow / self.nCol
        noise_img = np.random.normal(0, np.sqrt(var_img), [self.nRow * self.nCol, self.nBand])

        mixed = np.transpose(mixedpure + noise_img)  # channels*(rows*columns)
        #self.data = mixed
        return mixed, var_img

    def load_groundtruth(self, gt_loc):
        if (verbose):
            print('... Loading groundtruth')
        pkg_gt = sio.loadmat(gt_loc)
        self.groundtruth = pkg_gt['A'][0:200,:]  # (220, 4)
        self.num_gtendm = self.groundtruth.shape[1]  # 4

    def load_abf_true(self, abf_true_loc):
        if (verbose):
            print('... Loading true abundance')
        abf_gt = sio.loadmat(abf_true_loc)
        self.abf_gt = abf_gt['abf']

    def convert_2D(self, data):
        if (verbose):
            print('... Converting 3D data to 2D')
        self.nPixel = self.nRow * self.nCol
        data_2D = np.asmatrix(data.reshape((self.nRow * self.nCol, self.nBand))).T
        return data_2D

    # def extract_endmember(self):
    #     if (verbose):
    #         print('... Extracting endmembers')
    #     self.raw_endmembers = self.ee.extract_endmember()[0]     # 220*4 提取所得的端元光谱值

    def SAD(self, a, b):  # 计算光谱角距离
        if (verbose):
            print('... Applying SAD metric')
        [L, N] = a.shape  # L 波段数，N 端元数
        errRadians = np.zeros(N)
        b = np.asmatrix(b)
        for k in range(0, N):  # 逐端元
            tmp = np.asmatrix(np.reshape(a[:, k], (L, 1)))
            s1 = tmp.T
            s2 = b
            s1_norm = la.norm(s1)
            s2_norm = la.norm(s2)
            sum_s1_s2 = s1 * s2.T
            aux = sum_s1_s2 / (s1_norm * s2_norm)
            aux[aux > 1.0] = 1
            angle = math.acos(aux)
            errRadians[k] = angle
        return errRadians

    def SID(self, s1, s2):
        if (verbose):
            print('... Applying SID metric')
        [L, N] = s1.shape
        errRadians = np.zeros(N)
        s1 = np.asarray(s1)
        s2 = np.asarray(s2)
        for k in range(0, N):
            tmp = s1[:, k]
            p = (tmp / np.sum(tmp)) + np.spacing(1)
            q = (s2 / np.sum(s2)) + np.spacing(1)
            angle = np.sum(p.T * np.log(p / q) + q * np.log(q / p))

            if np.isnan(angle):
                errRadians[k] = 0
            else:
                errRadians[k] = angle

        return errRadians

    def AAD(self, a, b):
        if (verbose):
            print('... Applying AAD metric')
        #pdb.set_trace()
        [L, N] = a.shape  # L 端元数，N 像元数
        errRadians = np.zeros(L)
        b = np.asmatrix(b)
        for k in range(0, L):
            tmp = np.asmatrix(np.reshape(a[k, :], (1, N)))
            s1 = tmp
            s2 = b
            s1_norm = la.norm(s1)
            s2_norm = la.norm(s2)
            sum_s1_s2 = s1 * s2.T
            aux = sum_s1_s2 / (s1_norm * s2_norm)
            aux[aux > 1.0] = 1
            angle = math.acos(aux)
            errRadians[k] = angle
        return errRadians

    def AID(self, s1, s2):
        if (verbose):
            print('... Applying AID metric')
        [L, N] = s1.shape
        errRadians = np.zeros(L)
        s1 = np.asarray(s1)
        s2 = np.asarray(s2)
        for k in range(0, L):
            tmp = s1[k, :]
            p = (tmp / np.sum(tmp)) + np.spacing(1)
            q = (s2 / np.sum(s2)) + np.spacing(1)
            angle = np.sum(p.T * np.log(p / q) + q * np.log(q / p))

            if np.isnan(angle):
                errRadians[k] = 0
            else:
                errRadians[k] = angle

        return errRadians

    def best_sad_match(self):  # Best Estimated endmember for the groundtruth
        if (verbose):
            print('... Matching best endmember and groundtruth pair')
        sad = np.zeros((self.num_gtendm, self.p))    # self.num_gtendm:真实端元数；self.p：提取的端元数

        for i in range(self.raw_endmembers.shape[1]):  # 计算提取的每种端元与4种真实端元的光谱角距离
            sad[i, :] = self.SAD(self.raw_endmembers, self.groundtruth[:, i])    # 第i个真实端元值与提取的每种端元的光谱值

        idxs = [list(x) for x in np.argsort(sad, axis=1)]    # argsort函数返回的是数组值从小到大的索引值
        values = [list(x) for x in np.sort(sad, axis=1)]    # 排序
        pidxs = list(range(self.p))
        aux = []

        for i in range(self.num_gtendm):
            for j in range(self.p):
                aux.append([pidxs[i], idxs[i][j], values[i][j]])

        aux = sorted(aux, key=lambda x: x[2])
        new_idx = [0] * self.p
        new_value = [0] * self.p

        for i in range(self.p):
            a_idx, b_idx, c_value = aux[0]
            new_idx[a_idx] = b_idx
            new_value[a_idx] = c_value
            aux = [x for x in aux if x[0] != a_idx and x[1] != b_idx]

        return [new_idx, new_value]

    def best_aad_match(self):  # Best Estimated endmember for the groundtruth
        if (verbose):
            print('... Matching best endmember and groundtruth pair')
        aad = np.zeros((self.p, self.p))

        for i in range(self.abf_gt.shape[0]):  # 计算每种端元的光谱角距离&光谱信息距离
            aad[i, :] = self.AAD(self.S, self.abf_gt[i,:])

        idxs = [list(x) for x in np.argsort(aad, axis=1)]
        values = [list(x) for x in np.sort(aad, axis=1)]
        pidxs = list(range(self.nPixel))
        aux = []

        for i in range(self.p):
            for j in range(self.p):
                aux.append([pidxs[i], idxs[i][j], values[i][j]])

        aux = sorted(aux, key=lambda x: x[2])
        new_idx = [0] * self.p
        new_value = [0] * self.p

        for i in range(self.p):
            a_idx, b_idx, c_value = aux[0]
            new_idx[a_idx] = b_idx
            new_value[a_idx] = c_value
            aux = [x for x in aux if x[0] != a_idx and x[1] != b_idx]

        return [new_idx, new_value]

    def best_sid_match(self):
        if (verbose):
            print('... Matching best endmember and groundtruth pair')
        sid = np.zeros((self.num_gtendm, self.p))
        for i in range(self.raw_endmembers.shape[1]):
            sid[i, :] = self.SID(self.raw_endmembers, self.groundtruth[:, i])

        idxs = [list(x) for x in np.argsort(sid, axis=1)]
        values = [list(x) for x in np.sort(sid, axis=1)]
        pidxs = list(range(self.p))
        aux = []

        for i in range(self.num_gtendm):
            for j in range(self.p):
                aux.append([pidxs[i], idxs[i][j], values[i][j]])

        aux = sorted(aux, key=lambda x: x[2])
        new_idx = [0] * self.p
        new_value = [0] * self.p

        for i in range(self.p):
            a_idx, b_idx, c_value = aux[0]
            new_idx[a_idx] = b_idx
            new_value[a_idx] = c_value
            aux = [x for x in aux if x[0] != a_idx and x[1] != b_idx]

        return [new_idx, new_value]

    def best_aid_match(self):
        if (verbose):
            print('... Matching best endmember and groundtruth pair')
        aid = np.zeros((self.p, self.p))
        for i in range(self.S.shape[0]):
            aid[i, :] = self.AID(self.S, self.abf_gt[i, :])

        idxs = [list(x) for x in np.argsort(aid, axis=1)]
        values = [list(x) for x in np.sort(aid, axis=1)]
        pidxs = list(range(self.p))
        aux = []

        for i in range(self.p):
            for j in range(self.p):
                aux.append([pidxs[i], idxs[i][j], values[i][j]])

        aux = sorted(aux, key=lambda x: x[2])
        new_idx = [0] * self.p
        new_value = [0] * self.p

        for i in range(self.p):
            a_idx, b_idx, c_value = aux[0]
            new_idx[a_idx] = b_idx
            new_value[a_idx] = c_value
            aux = [x for x in aux if x[0] != a_idx and x[1] != b_idx]

        return [new_idx, new_value]

    def best_run(self, mrun, num):
        if (verbose):
            print('... Starting Monte Carlo set of runs')
        sad_min_run = 9999
        sad_min_values = None
        sad_min_idx = None
        sad_min_A = None
        sad_min_S = None

        aad_min_run = 9999
        aad_min_values = None
        aad_min_idx = None
        aad_min_A = None
        aad_min_S = None

        sad_max_run = -9999
        sad_max_values = None
        sad_max_idx = None
        sad_max_A = None
        sad_max_S = None

        aad_max_run = -9999
        aad_max_values = None
        aad_max_idx = None
        aad_max_A = None
        aad_max_S = None

        self.sad_all_runs_value = np.zeros((mrun, self.p))
        self.aad_all_runs_value = np.zeros((mrun, self.p))

        self.sad_S_img = np.zeros((self.num_gtendm, self.nRow, self.nCol))
        self.abf_gt_img = np.zeros((self.num_gtendm, self.nRow, self.nCol))

        sid_min_run = 9999
        sid_min_values = None
        sid_min_idx = None
        sid_min_A = None
        sid_min_S = None

        aid_min_run = 9999
        aid_min_values = None
        aid_min_idx = None
        aid_min_A = None
        aid_min_S = None

        sid_max_run = -9999
        sid_max_values = None
        sid_max_idx = None
        sid_max_A = None
        sid_max_S = None

        aid_max_run = -9999
        aid_max_values = None
        aid_max_idx = None
        aid_max_A = None
        aid_max_S = None

        self.sid_all_runs_value = np.zeros((mrun, self.p))
        self.aid_all_runs_value = np.zeros((mrun, self.p))

        self.time_runs = []
        for i in range(mrun):
            mixed_loc = "./results/data_out_" + str(i+1) + "/" + str(g) + "_mixed.mat"
            self.data = sio.loadmat(mixed_loc)['mixed']
            A_loc = "/results/data_out_" + str(i+1) + "/endmember_" + str(g) + ".mat"
            S_loc = "/results/data_out_" + str(i+1) + "/abundance_" + str(g) + ".mat"
            A_all = sio.loadmat(A_loc)['A']
            S_all = sio.loadmat(S_loc)['S']
            self.raw_endmembers = A_all[num, :, :]
            self.S = S_all[num, :, :]

            [sad_idx, sad_value] = self.best_sad_match()
            [sid_idx, sid_value] = self.best_sid_match()

            [aad_idx, aad_value] = self.best_aad_match()
            [aid_idx, aid_value] = self.best_aid_match()

            self.sad_all_runs_value[i, :] = sad_value
            self.sid_all_runs_value[i, :] = sid_value

            self.aad_all_runs_value[i, :] = aad_value
            self.aid_all_runs_value[i, :] = aid_value

            if (np.mean(sad_value) <= sad_min_run):
                #pdb.set_trace()
                sad_min_run = np.mean(sad_value)
                sad_min_values = sad_value
                sad_min_idx = sad_idx
                sad_min_A = self.raw_endmembers[:, sad_idx]
                sad_min_S = self.S[sad_idx, :]

            if (np.mean(aad_value) <= aad_min_run):
                aad_min_run = np.mean(aad_value)
                aad_min_values = aad_value
                aad_min_idx = aad_idx
                aad_min_A = self.raw_endmembers[:, aad_idx]
                aad_min_S = self.S[aad_idx, :]

            if (np.mean(sad_value) >= sad_max_run):
                sad_max_run = np.mean(sad_value)
                sad_max_values = sad_value
                sad_max_idx = sad_idx
                sad_max_A = self.raw_endmembers[:, sad_idx]
                sad_max_S = self.S[sad_idx, :]

            if (np.mean(aad_value) >= aad_max_run):
                aad_max_run = np.mean(aad_value)
                aad_max_values = aad_value
                aad_max_idx = aad_idx
                aad_max_A = self.raw_endmembers[:, aad_idx]
                aad_max_S = self.S[aad_idx, :]

            if (np.mean(sid_value) <= sid_min_run):
                sid_min_run = np.mean(sid_value)
                sid_min_values = sid_value
                sid_min_idx = sid_idx
                sid_min_A = self.raw_endmembers[:, sid_idx]
                sid_min_S = self.S[sid_idx, :]

            if (np.mean(aid_value) <= aid_min_run):
                aid_min_run = np.mean(aid_value)
                aid_min_values = aid_value
                aid_min_idx = aid_idx
                aid_min_A = self.raw_endmembers[:, aid_idx]
                aid_min_S = self.S[aid_idx, :]

            if (np.mean(sid_value) <= sid_max_run):
                sid_max_run = np.mean(sid_value)
                sid_max_values = sid_value
                sid_max_idx = sid_idx
                sid_max_A = self.raw_endmembers[:, sid_idx]
                sid_max_S =self.S[sid_idx, :]

            if (np.mean(aid_value) <= aid_max_run):
                aid_max_run = np.mean(aid_value)
                aid_max_values = aid_value
                aid_max_idx = aid_idx
                aid_max_A = self.raw_endmembers[:, aid_idx]
                aid_max_S = self.S[aid_idx, :]

        self.sad_em_max = sad_max_A
        self.sad_em_min = sad_min_A
        self.sad_ab_max = sad_max_S
        self.sad_ab_min = sad_min_S

        self.aad_em_max = aad_max_A
        self.aad_em_min = aad_min_A
        self.aad_ab_max = aad_max_S
        self.aad_ab_min = aad_min_S

        self.sad_values_max = sad_max_values
        self.sad_values_min = sad_min_values

        self.aad_values_max = aad_max_values
        self.aad_values_min = aad_min_values

        self.sad_max = sad_max_run
        self.sad_min = sad_min_run

        self.aad_max = aad_max_run
        self.aad_min = aad_min_run

        self.sad_mean = np.mean(self.sad_all_runs_value, axis=0)
        self.sad_var = np.var(self.sad_all_runs_value, axis=0)
        self.sad_std = np.std(self.sad_all_runs_value, axis=0)

        self.aad_mean = np.mean(self.aad_all_runs_value, axis=0)
        self.aad_var = np.var(self.aad_all_runs_value, axis=0)
        self.aad_std = np.std(self.aad_all_runs_value, axis=0)

        self.sid_em_max = sid_max_A
        self.sid_em_min = sid_min_A
        self.sid_ab_max = sid_max_S
        self.sid_ab_min = sid_min_S

        self.aid_em_max = aid_max_A
        self.aid_em_min = aid_min_A
        self.aid_ab_max = aid_max_S
        self.aid_ab_min = aid_min_S

        self.sid_values_max = sid_max_values
        self.sid_values_min = sid_min_values

        self.aid_values_max = aid_max_values
        self.aid_values_min = aid_min_values

        self.sid_max = sid_max_run
        self.sid_min = sid_max_run

        self.aid_max = aid_max_run
        self.aid_min = aid_max_run

        self.sid_mean = np.mean(self.sid_all_runs_value, axis=0)
        self.sid_var = np.var(self.sid_all_runs_value, axis=0)
        self.sid_std = np.std(self.sid_all_runs_value, axis=0)

        self.aid_mean = np.mean(self.aid_all_runs_value, axis=0)
        self.aid_var = np.var(self.aid_all_runs_value, axis=0)
        self.aid_std = np.std(self.aid_all_runs_value, axis=0)

        #pdb.set_trace()
        self.sad_ssim = 1-ssim(self.sad_ab_min, self.abf_gt, win_size=3, multichannel=True)
        self.sad_em_m = np.asmatrix(self.sad_em_min)
        self.sad_ab_m = np.asmatrix(self.sad_ab_min)
        self.sad_mse = mse(self.data, self.sad_em_m * self.sad_ab_m)

        self.sid_ssim = 1-ssim(self.sid_ab_min, self.abf_gt, win_size=3, multichannel=True)
        self.sid_em_m = np.asmatrix(self.sid_em_min)
        self.sid_ab_m = np.asmatrix(self.sid_ab_min)
        self.sid_mse = mse(self.data, self.sid_em_m * self.sid_ab_m)

        self.aad_ssim = 1-ssim(self.aad_ab_min, self.abf_gt, win_size=3, multichannel=True)
        self.aad_em_m = np.asmatrix(self.aad_em_min)
        self.aad_ab_m = np.asmatrix(self.aad_ab_min)
        self.aad_mse = mse(self.data, self.aad_em_m * self.aad_ab_m)

        self.aid_ssim = 1-ssim(self.aid_ab_min, self.abf_gt, win_size=3, multichannel=True)
        self.aid_em_m = np.asmatrix(self.aid_em_min)
        self.aid_ab_m = np.asmatrix(self.aid_ab_min)
        self.aid_mse = mse(self.data, self.aid_em_m * self.aid_ab_m)

        self.sad_S_img = np.reshape(self.sad_ab_min, [self.num_gtendm, self.nRow, self.nCol])
        self.abf_gt_img = np.reshape(self.abf_gt, [self.num_gtendm, self.nRow, self.nCol])

def run():
    endmember_names = ['A', 'B', 'C', 'D']

    print("AEsmm")
    aesmm = DEMO([data_loc, gt_loc, num_endm, 'AEsmm'], verbose)
    aesmm.best_run(mrun, 0)
    print("PPI")
    ppi = DEMO([data_loc, gt_loc, num_endm, 'PPI', nSkewers, initSkewers], verbose)
    ppi.best_run(mrun, 1)
    print("NFINDR")
    nfindr = DEMO([data_loc, gt_loc, num_endm, 'NFINDR', maxit], verbose)
    nfindr.best_run(mrun, 2)
    print("VCA")
    vca = DEMO([data_loc, gt_loc, num_endm, 'VCA'], verbose)
    vca.best_run(mrun, 3)
    print("KPmeans")
    kpmeans = DEMO([data_loc, gt_loc, num_endm, 'uDAs'], verbose)
    kpmeans.best_run(mrun, 4)

    algo = [ppi, nfindr, vca, kpmeans, aesmm]

    tab1_sad = pd.DataFrame()
    tab1_sad['Endmembers'] = endmember_names
    tab1_sad.set_index('Endmembers', inplace=True)

    tab1_aad = pd.DataFrame()
    tab1_aad['Abundance'] = endmember_names
    tab1_aad.set_index('Abundance', inplace=True)

    tab2_sid = pd.DataFrame()
    tab2_sid['Endmembers'] = endmember_names
    tab2_sid.set_index('Endmembers', inplace=True)

    tab2_aid = pd.DataFrame()
    tab2_aid['Abundance'] = endmember_names
    tab2_aid.set_index('Abundance', inplace=True)

    tab4_sad_stats = pd.DataFrame()
    tab4_sad_stats['Statistics'] = ['_Mean_', '_Std_', '_1-Ssim_', '_Mse_']
    tab4_sad_stats.set_index('Statistics', inplace=True)

    tab4_aad_stats = pd.DataFrame()
    tab4_aad_stats['Statistics'] = ['_Mean_', '_Std_', '_1-Ssim_', '_Mse_']
    tab4_aad_stats.set_index('Statistics', inplace=True)

    tab5_sid_stats = pd.DataFrame()
    tab5_sid_stats['Statistics'] = ['_Mean_', '_Std_', '_1-Ssim_', '_Mse_']
    tab5_sid_stats.set_index('Statistics', inplace=True)

    tab5_aid_stats = pd.DataFrame()
    tab5_aid_stats['Statistics'] = ['_Mean_', '_Std_', '_1-Ssim_', '_Mse_']
    tab5_aid_stats.set_index('Statistics', inplace=True)


    for l in algo:
        tab1_sad[l.name] = l.sad_values_min
        tab4_sad_stats[l.name] = [np.mean(l.sad_mean), np.mean(l.sad_std), l.sad_ssim, l.sad_mse]

        tab2_sid[l.name] = l.sid_values_min
        tab5_sid_stats[l.name] = [np.mean(l.sid_mean), np.mean(l.sid_std), l.sid_ssim, l.sid_mse]

        tab1_aad[l.name] = l.aad_values_min
        tab4_aad_stats[l.name] = [np.mean(l.aad_mean), np.mean(l.aad_std), l.aad_ssim, l.aad_mse]

        tab2_aid[l.name] = l.aid_values_min
        tab5_aid_stats[l.name] = [np.mean(l.aid_mean), np.mean(l.aid_std), l.aid_ssim, l.aid_mse]

    file.write(
        '### Comparison between the ground-truth and extracted endmembers using PPI, N-FINDR, VCA using SAD for the simulate Dataset.\n\n')
    table_fancy = tabulate(tab1_sad, tablefmt="pipe", floatfmt=".4f", headers="keys")
    for idx, mi in enumerate([min([k for k in t.split() if k != '|'][1:]) for t in table_fancy.split('\n')[2:]]):
        line = table_fancy.split('\n')[2:][idx]
        table_fancy = table_fancy.replace(line, line.replace(' ' + mi + ' ', ' **' + mi + '** '))
    file.write(table_fancy + '\n\n')

    file.write('### SAD Statistics for Cuprite Dataset. \n\n')
    table_fancy = tabulate(tab4_sad_stats, tablefmt="pipe", floatfmt=".4f", headers="keys")
    for idx, mi in enumerate([min([k for k in t.split() if k != '|'][1:]) for t in table_fancy.split('\n')[2:]]):
        line = table_fancy.split('\n')[2:][idx]
        table_fancy = table_fancy.replace(line, line.replace(' ' + mi + ' ', ' **' + mi + '** '))
    file.write(table_fancy + '\n\n')

    file.write(
        '### Comparison between the ground-truth and extracted endmembers using PPI, N-FINDR, VCA, using AAD for the simulate Dataset.\n\n')
    table_fancy = tabulate(tab1_aad, tablefmt="pipe", floatfmt=".4f", headers="keys")
    for idx, mi in enumerate([min([k for k in t.split() if k != '|'][1:]) for t in table_fancy.split('\n')[2:]]):
        line = table_fancy.split('\n')[2:][idx]
        table_fancy = table_fancy.replace(line, line.replace(' ' + mi + ' ', ' **' + mi + '** '))
    file.write(table_fancy + '\n\n')
    file.write('### AAD Statistics for Cuprite Dataset. \n\n')
    table_fancy = tabulate(tab4_aad_stats, tablefmt="pipe", floatfmt=".4f", headers="keys")
    for idx, mi in enumerate([min([k for k in t.split() if k != '|'][1:]) for t in table_fancy.split('\n')[2:]]):
        line = table_fancy.split('\n')[2:][idx]
        table_fancy = table_fancy.replace(line, line.replace(' ' + mi + ' ', ' **' + mi + '** '))
    file.write(table_fancy + '\n\n')

    # file.write(tabulate(tab1_sad, tablefmt="pipe", headers="keys")+'\n\n')
    file.write(
        '### Comparison between the ground-truth Laboratory Reflectances and extracted endmembers using PPI, N-FINDR, VCA using SID for the Cuprite Dataset.\n\n')
    # file.write(tabulate(tab2_sid, tablefmt="pipe", headers="keys")+'\n\n')

    table_fancy = tabulate(tab2_sid, tablefmt="pipe", floatfmt=".4f", headers="keys")
    for idx, mi in enumerate([min([k for k in t.split() if k != '|'][1:]) for t in table_fancy.split('\n')[2:]]):
        line = table_fancy.split('\n')[2:][idx]
        table_fancy = table_fancy.replace(line, line.replace(' ' + mi + ' ', ' **' + mi + '** '))
    file.write(table_fancy + '\n\n')

    file.write('### SID Statistics for Simulated Data. \n\n')
    table_fancy = tabulate(tab5_sid_stats, tablefmt="pipe", floatfmt=".4f", headers="keys")
    for idx, mi in enumerate([min([k for k in t.split() if k != '|'][1:]) for t in table_fancy.split('\n')[2:]]):
        line = table_fancy.split('\n')[2:][idx]
        table_fancy = table_fancy.replace(line, line.replace(' ' + mi + ' ', ' **' + mi + '** '))
    file.write(table_fancy + '\n\n')

    # file.write(tabulate(tab1_sad, tablefmt="pipe", headers="keys")+'\n\n')
    file.write(
        '### Comparison between the ground-truth Laboratory Reflectances and extracted endmembers using PPI, N-FINDR, VCA using AID for the Cuprite Dataset.\n\n')
    table_fancy = tabulate(tab2_aid, tablefmt="pipe", floatfmt=".4f", headers="keys")
    for idx, mi in enumerate([min([k for k in t.split() if k != '|'][1:]) for t in table_fancy.split('\n')[2:]]):
        line = table_fancy.split('\n')[2:][idx]
        table_fancy = table_fancy.replace(line, line.replace(' ' + mi + ' ', ' **' + mi + '** '))
    file.write(table_fancy + '\n\n')

    file.write('### AID Statistics for Simulated Data. \n\n')
    table_fancy = tabulate(tab5_aid_stats, tablefmt="pipe", floatfmt=".4f", headers="keys")
    for idx, mi in enumerate([min([k for k in t.split() if k != '|'][1:]) for t in table_fancy.split('\n')[2:]]):
        line = table_fancy.split('\n')[2:][idx]
        table_fancy = table_fancy.replace(line, line.replace(' ' + mi + ' ', ' **' + mi + '** '))
    file.write(table_fancy + '\n\n')

    # endmember = []
    # abundance = []
    # for idx, k in enumerate(algo):
    #     endmember.append(k.sad_em_min)
    #     abundance.append(k.sad_ab_min)
    # snr_w = str(g)
    # sio.savemat('./results/data_out_all/endmember.mat' + '_' + snr_w, {'A': endmember})
    # sio.savemat('./results/data_out_all/abundance.mat' + '_' + snr_w, {'S': abundance})
    snr_w = str(g)
    for idx, k in enumerate(algo):
        sio.savemat('./results/data_out_all/SAD.mat' + '_' + snr_w + '_' + k.name, {'SAD': k.sad_all_runs_value})
        sio.savemat('./results/data_out_all/SID.mat' + '_' + snr_w + '_' + k.name, {'SID': k.sid_all_runs_value})
        sio.savemat('./results/data_out_all/AAD.mat' + '_' + snr_w + '_' + k.name, {'AAD': k.aad_all_runs_value})
        sio.savemat('./results/data_out_all/AID.mat' + '_' + snr_w + '_' + k.name, {'AID': k.aid_all_runs_value})

    for i in range(0, 4):  # 绘制端元对比图
        fig = plt.figure()
        plt.plot(vca.groundtruth[:, i], label='USGS Library')

        for idx, k in enumerate(algo):
            new_endmember = np.empty((k.groundtruth.shape[0], k.p))
            new_endmember[:] = np.nan
            new_endmember[:, :] = k.sad_em_min

            plt.plot(new_endmember[:, i], label=k.name)

        plt.title(endmember_names[i])

        plt.legend()
        plt.xlabel("wavelength (um)")
        plt.ylabel('reflectance (%)')
        plt.tight_layout()


        plt.savefig('./results/IMG_all/SNR=' + snr_w + '_' + endmember_names[i] + '_Endmember.png', format='png', dpi=200)
        file.write('![alt text](./IMG_all/SNR=' + snr_w + '_' + endmember_names[i] + '_Endmember.png)\n\n')

        # 绘制丰度对比图
        new_abundance = []
        new_abundance.append(np.reshape(algo[0].sad_S_img[i, :, :],[1,nR,nC]))
        new_abundance.append(np.reshape(algo[1].sad_S_img[i, :, :],[1,nR,nC]))
        new_abundance.append(np.reshape(algo[2].sad_S_img[i, :, :],[1,nR,nC]))
        new_abundance.append(np.reshape(algo[3].sad_S_img[i, :, :],[1,nR,nC]))
        new_abundance.append(np.reshape(algo[4].sad_S_img[i, :, :],[1,nR,nC]))
        new_abundance.append(np.reshape(algo[0].abf_gt_img[i, :, :],[1,nR,nC]))
        for idx, k in enumerate(algo):
            abundance_one = np.reshape(algo[idx].sad_S_img[i, :, :],[1,1,nR,nC])
            plt_one = plot_image_grid(abundance_one, factor=8, nrow=1)
            plt_one.title(endmember_names[i] + '_'+ k.name)
            plt_one.tight_layout()
            plt_one.savefig('./results/IMG_spilt/SNR=' + snr_w + '_' + endmember_names[i] + '_' + k.name + '.png', format='png', dpi=200, bbox_inches = 'tight')
        abundance_true = np.reshape(algo[0].abf_gt_img[i, :, :], [1, 1, nR, nC])
        plt_one = plot_image_grid(abundance_true, factor=8, nrow=1)
        plt_one.title(endmember_names[i] + '_' + k.name)
        plt_one.tight_layout()
        plt_one.savefig('./results/IMG_spilt/SNR=' + snr_w + '_' + endmember_names[i] + '_' + 'TRUE' + '.png',
                        format='png', dpi=200, bbox_inches='tight')


        plt_s = plot_image_grid(new_abundance, factor=8, nrow=6)
        plt_s.title(endmember_names[i])
        plt_s.tight_layout()
        plt_s.savefig('./results/IMG_all/SNR=' + snr_w + '_' + endmember_names[i] + '_Abundance.png', format='png', dpi=200)
        file.write('![alt text](./IMG_all/SNR=' + snr_w + '_' + endmember_names[i] + '_Abundance.png)\n\n')



if __name__ == '__main__':
    data_loc = "./DATA/dip_data/unmixing3/mixed.mat"
    gt_loc = "./DATA/dip_data/unmixing3/OtherInfo/A.mat"
    abf_true_loc = "./DATA/dip_data/unmixing3/OtherInfo/abf.mat"

    debug = True
    verbose = True
    num_endm = 4    # 混合影像数据的端元数
    nSkewers = 1000    # PPI的投影向量个数
    initSkewers = None    # PPI的初始投影向量个数
    maxit = 3 * num_endm    # N-FINDR方法的最大迭代次数
    # maxit_mvc = 50
    SNR_noise = [10,20,30,40]
    mrun = 10 # 每种方法的迭代次数
    interation = 10 # EM iteration number of traditional methods for A and S estimation 20

    # # for AEsmm
    iterEM = 50
    iters = 20
    INPUT = 'noise'
    pad = 'reflection'
    OPT_OVER = 'net'
    LR = 0.00025
    tv_weight = 0.0
    OPTIMIZER = 'adam'
    NET_TYPE = 'skip'
    # # # #

    file = open("./results/demo_all.md", "w")
    file.write("# Comparison of Unmixing method" + "\n\n")
    file.write('## Envirionment Setup: \n\n')
    file.write('Monte Carlo runs: %s \n\n' % mrun)  # 每种方法的迭代次数
    file.write('Number of endmembers to estimate: %s \n\n' % num_endm)  # 提取的端元数
    file.write('Number of skewers (PPI): %s \n\n' % nSkewers)  # PPI方法的投影向量数
    file.write('Maximum number of iterations (N-FINDR): %s \n\n' % maxit)  # N-FINDR方法的最大迭代次数
    # file.write('Maximum number of iterations (MVC-NMF): %s \n\n' % maxit_mvc)  # MVC-NMF方法的最大迭代次数
    file.write('Maximum number of iterations (nnls): %s \n\n' % interation)  # 丰度估计和端元更新的迭代次数

    if debug:
        for s in range(0, len(SNR_noise)):
            g = SNR_noise[s]
            file.write('## SNR = : %s \n\n' % g)  # SNR的大小
            run()    # 运行

    file.close()