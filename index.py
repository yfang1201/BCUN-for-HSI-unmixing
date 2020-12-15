#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : xian
# @Time    : 2019/10/22 22:40

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

def load_data(data_loc):
    if (verbose):
        print('... Loading data')

    pkg_data = sio.loadmat(data_loc)
    return pkg_data

def load_groundtruth(gt_loc):
    if (verbose):
        print('... Loading groundtruth')
    pkg_gt = sio.loadmat(gt_loc)
    groundtruth = pkg_gt['A'][0:200,:]
    num_gtendm = groundtruth.shape[1]
    return  groundtruth, num_gtendm

def load_abf_true(abf_true_loc):
    if (verbose):
        print('... Loading true abundance')
    abf_gt = sio.loadmat(abf_true_loc)
    abf_gt = abf_gt['abf']
    return abf_gt

def SAD(a, b):  # 计算光谱角距离
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

def SID(s1, s2):
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

def AAD(a, b):
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

def AID(s1, s2):
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

def best_sad_match(raw_endmembers,groundtruth,num_gtendm,p):  # Best Estimated endmember for the groundtruth
    if (verbose):
        print('... Matching best endmember and groundtruth pair')
    sad = np.zeros((num_gtendm, p))

    for i in range(raw_endmembers.shape[1]):
        sad[i, :] = SAD(raw_endmembers, groundtruth[:, i])

    idxs = [list(x) for x in np.argsort(sad, axis=1)]
    values = [list(x) for x in np.sort(sad, axis=1)]
    pidxs = list(range(p))
    aux = []

    for i in range(num_gtendm):
        for j in range(p):
            aux.append([pidxs[i], idxs[i][j], values[i][j]])

    aux = sorted(aux, key=lambda x: x[2])
    new_idx = [0] * p
    new_value = [0] * p

    for i in range(p):
        a_idx, b_idx, c_value = aux[0]
        new_idx[a_idx] = b_idx
        new_value[a_idx] = c_value
        aux = [x for x in aux if x[0] != a_idx and x[1] != b_idx]

    return [new_idx, new_value]

def best_aad_match(S, abf_gt, p, nPixel):  # Best Estimated endmember for the groundtruth
    if (verbose):
        print('... Matching best endmember and groundtruth pair')
    aad = np.zeros((p, p))

    for i in range(abf_gt.shape[0]):  # 计算每种端元的光谱角距离&光谱信息距离
        aad[i, :] = AAD(S, abf_gt[i,:])

    idxs = [list(x) for x in np.argsort(aad, axis=1)]
    values = [list(x) for x in np.sort(aad, axis=1)]
    pidxs = list(range(nPixel))
    aux = []

    for i in range(p):
        for j in range(p):
            aux.append([pidxs[i], idxs[i][j], values[i][j]])

    aux = sorted(aux, key=lambda x: x[2])
    new_idx = [0] * p
    new_value = [0] * p

    for i in range(p):
        a_idx, b_idx, c_value = aux[0]
        new_idx[a_idx] = b_idx
        new_value[a_idx] = c_value
        aux = [x for x in aux if x[0] != a_idx and x[1] != b_idx]

    return [new_idx, new_value]

def best_sid_match(raw_endmembers,groundtruth,num_gtendm,p):
    if (verbose):
        print('... Matching best endmember and groundtruth pair')
    sid = np.zeros((num_gtendm, p))
    for i in range(raw_endmembers.shape[1]):
        sid[i, :] = SID(raw_endmembers, groundtruth[:, i])

    idxs = [list(x) for x in np.argsort(sid, axis=1)]
    values = [list(x) for x in np.sort(sid, axis=1)]
    pidxs = list(range(p))
    aux = []

    for i in range(num_gtendm):
        for j in range(p):
            aux.append([pidxs[i], idxs[i][j], values[i][j]])

    aux = sorted(aux, key=lambda x: x[2])
    new_idx = [0] * p
    new_value = [0] * p

    for i in range(p):
        a_idx, b_idx, c_value = aux[0]
        new_idx[a_idx] = b_idx
        new_value[a_idx] = c_value
        aux = [x for x in aux if x[0] != a_idx and x[1] != b_idx]

    return [new_idx, new_value]

def best_aid_match(S, abf_gt, p):
    if (verbose):
        print('... Matching best endmember and groundtruth pair')
    aid = np.zeros((p, p))
    for i in range(S.shape[0]):
        aid[i, :] = AID(S, abf_gt[i, :])

    idxs = [list(x) for x in np.argsort(aid, axis=1)]
    values = [list(x) for x in np.sort(aid, axis=1)]
    pidxs = list(range(p))
    aux = []

    for i in range(p):
        for j in range(p):
            aux.append([pidxs[i], idxs[i][j], values[i][j]])

    aux = sorted(aux, key=lambda x: x[2])
    new_idx = [0] * p
    new_value = [0] * p

    for i in range(p):
        a_idx, b_idx, c_value = aux[0]
        new_idx[a_idx] = b_idx
        new_value[a_idx] = c_value
        aux = [x for x in aux if x[0] != a_idx and x[1] != b_idx]

    return [new_idx, new_value]



if __name__ == '__main__':
    # gt_loc = "./DATA/dip_data/unmixing3/OtherInfo/A.mat"
    # abf_true_loc = "./DATA/dip_data/unmixing3/OtherInfo/abf.mat"
    # A_true = load_data(gt_loc)['A']
    # S_true = load_data(abf_true_loc)['abf']

    debug = True
    verbose = True
    num_endm = 4    # 混合影像数据的端元数
    SNR_noise = [10,20,30,40]
    mrun = 10 # 每种方法的迭代次数
    p = 4

    file = open("./mat_results/my_all.md", "w")
    file.write("# Comparison of Unmixing method" + "\n\n")
    file.write('## Envirionment Setup: \n\n')
    file.write('Monte Carlo runs: %s \n\n' % mrun)
    file.write('Number of endmembers to estimate: %s \n\n' % num_endm)

    sad_all_runs_value = np.zeros((mrun, p))
    sid_all_runs_value = np.zeros((mrun, p))
    aad_all_runs_value = np.zeros((mrun, p))
    aid_all_runs_value = np.zeros((mrun, p))

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

    data_loc = "./DATA/dip_data/unmixing3/mixed.mat"
    gt_loc = "./DATA/dip_data/unmixing3/OtherInfo/A.mat"
    abf_true_loc = "./DATA/dip_data/unmixing3/OtherInfo/abf.mat"
    mixed = load_data(data_loc)['mixed']
    nRow = mixed.shape[0]
    nCol = mixed.shape[1]
    nBand = mixed.shape[2]
    nPixel = nRow * nCol
    A_true = load_groundtruth(gt_loc)
    S_true = load_abf_true(abf_true_loc)

    if debug:
        for s in range(0, len(SNR_noise)):
            g = SNR_noise[s]
            for j in range(0, 5):
                for i in range(1, mrun + 1):
                    A_loc = "./mat_results/" + str(i) + "/A" + str(g) + ".mat"
                    S_loc = "./mat_results/" + str(i) + "/S" + str(g) + ".mat"

                    A_all = load_data(A_loc)['A']
                    S_all = load_data(S_loc)['S']

                    # A_true = np.squeeze(A_all[5, :, :], 0)
                    # S_true = np.squeeze(S_all[5, :, :], 0)

                    raw_endmembers = np.squeeze(A_all[j, :, :], 0)
                    raw_abundance = np.squeeze(S_all[j, :, :], 0)

                    [sad_idx, sad_value] = best_sad_match(raw_endmembers, A_true, p, p)
                    [sid_idx, sid_value] = best_sid_match(raw_endmembers, A_true,p,p)
                    [aad_idx, aad_value] = best_aad_match(raw_abundance, S_true, p, nPixel)
                    [aid_idx, aid_value] = best_aid_match(raw_abundance, S_true, p)

                    sad_all_runs_value[i, :] = sad_value
                    sid_all_runs_value[i, :] = sid_value
                    aad_all_runs_value[i, :] = aad_value
                    aid_all_runs_value[i, :] = aid_value

                    if (np.mean(sad_value) <= sad_min_run):
                        sad_min_run = np.mean(sad_value)
                        sad_min_values = sad_value
                        sad_min_idx = sad_idx
                        sad_min_A = raw_endmembers[:, sad_idx]
                        sad_min_S = raw_abundance[sad_idx, :]

                    if (np.mean(aad_value) <= aad_min_run):
                        aad_min_run = np.mean(aad_value)
                        aad_min_values = aad_value
                        aad_min_idx = aad_idx
                        aad_min_A = raw_endmembers[:, aad_idx]
                        aad_min_S = raw_abundance[aad_idx, :]

                    if (np.mean(sad_value) >= sad_max_run):
                        sad_max_run = np.mean(sad_value)
                        sad_max_values = sad_value
                        sad_max_idx = sad_idx
                        sad_max_A = raw_endmembers[:, sad_idx]
                        sad_max_S = raw_abundance[sad_idx, :]

                    if (np.mean(aad_value) >= aad_max_run):
                        aad_max_run = np.mean(aad_value)
                        aad_max_values = aad_value
                        aad_max_idx = aad_idx
                        aad_max_A = raw_endmembers[:, aad_idx]
                        aad_max_S = raw_abundance[aad_idx, :]

                    if (np.mean(sid_value) <= sid_min_run):
                        sid_min_run = np.mean(sid_value)
                        sid_min_values = sid_value
                        sid_min_idx = sid_idx
                        sid_min_A = raw_endmembers[:, sid_idx]
                        sid_min_S = raw_abundance[sid_idx, :]

                    if (np.mean(aid_value) <= aid_min_run):
                        aid_min_run = np.mean(aid_value)
                        aid_min_values = aid_value
                        aid_min_idx = aid_idx
                        aid_min_A = raw_endmembers[:, aid_idx]
                        aid_min_S = raw_abundance[aid_idx, :]

                    if (np.mean(sid_value) <= sid_max_run):
                        sid_max_run = np.mean(sid_value)
                        sid_max_values = sid_value
                        sid_max_idx = sid_idx
                        sid_max_A = raw_endmembers[:, sid_idx]
                        sid_max_S = raw_abundance[sid_idx, :]

                    if (np.mean(aid_value) <= aid_max_run):
                        aid_max_run = np.mean(aid_value)
                        aid_max_values = aid_value
                        aid_max_idx = aid_idx
                        aid_max_A = raw_endmembers[:, aid_idx]
                        aid_max_S = raw_abundance[aid_idx, :]

                sad_em_max = sad_max_A
                sad_em_min = sad_min_A
                sad_ab_max = sad_max_S
                sad_ab_min = sad_min_S

                sad_values_max = sad_max_values
                sad_values_min = sad_min_values

                sad_max = sad_max_run
                sad_min = sad_min_run

                sad_mean = np.mean(sad_all_runs_value, axis=0)
                sad_var = np.var(sad_all_runs_value, axis=0)
                sad_std = np.std(sad_all_runs_value, axis=0)

                sad_ssim = 1 - ssim(sad_ab_min, S_true, win_size=3, multichannel=True)
                sad_em_m = np.asmatrix(sad_em_min)
                sad_ab_m = np.asmatrix(sad_ab_min)
                sad_mse = mse(mixed, sad_em_m * sad_ab_m)

                aad_em_max = aad_max_A
                aad_em_min = aad_min_A
                aad_ab_max = aad_max_S
                aad_ab_min = aad_min_S

                aad_values_max = aad_max_values
                aad_values_min = aad_min_values

                aad_max = aad_max_run
                aad_min = aad_min_run

                aad_mean = np.mean(aad_all_runs_value, axis=0)
                aad_var = np.var(aad_all_runs_value, axis=0)
                aad_std = np.std(aad_all_runs_value, axis=0)

                aad_ssim = 1 - ssim(aad_ab_min, S_true, win_size=3, multichannel=True)
                aad_em_m = np.asmatrix(aad_em_min)
                aad_ab_m = np.asmatrix(aad_ab_min)
                aad_mse = mse(mixed, aad_em_m * aad_ab_m)

                sid_em_max = sid_max_A
                sid_em_min = sid_min_A
                sid_ab_max = sid_max_S
                sid_ab_min = sid_min_S

                sid_values_max = sid_max_values
                sid_values_min = sid_min_values

                sid_max = sid_max_run
                sid_min = sid_max_run

                sid_ssim = 1 - ssim(sid_ab_min, S_true, win_size=3, multichannel=True)
                sid_em_m = np.asmatrix(sid_em_min)
                sid_ab_m = np.asmatrix(sid_ab_min)
                sid_mse = mse(mixed, sid_em_m * sid_ab_m)

                sid_mean = np.mean(sid_all_runs_value, axis=0)
                sid_var = np.var(sid_all_runs_value, axis=0)
                sid_std = np.std(sid_all_runs_value, axis=0)

                aid_em_max = aid_max_A
                aid_em_min = aid_min_A
                aid_ab_max = aid_max_S
                aid_ab_min = aid_min_S

                aid_values_max = aid_max_values
                aid_values_min = aid_min_values

                aid_max = aid_max_run
                aid_min = aid_max_run

                aid_mean = np.mean(aid_all_runs_value, axis=0)
                aid_var = np.var(aid_all_runs_value, axis=0)
                aid_std = np.std(aid_all_runs_value, axis=0)

                aid_ssim = 1 - ssim(aid_ab_min, S_true, win_size=3, multichannel=True)
                aid_em_m = np.asmatrix(aid_em_min)
                aid_ab_m = np.asmatrix(aid_ab_min)
                aid_mse = mse(mixed, aid_em_m * aid_ab_m)

                sad_S_img = np.reshape(sad_ab_min, [p, nRow, nCol])
                abf_gt_img = np.reshape(S_true, [p, nRow, nCol])


