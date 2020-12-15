#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Author  : xian
# @Time    : 2019/9/16 22:34


import scipy.io as sio
import numpy as np

def load_data(data_loc):
    if (verbose):
        print('... Loading data')

    pkg_data = sio.loadmat(data_loc)
    data_clean = pkg_data['mixed'][:, :, 0:200]
    return data_clean


def addnoise(img, SNR):
    nRow = img.shape[0]
    nCol = img.shape[1]
    nBand = img.shape[2]
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
    mixedpure = np.reshape(img, [nRow * nCol, nBand])

    snr = SNR + scal_nor_cen_snr_noi
    var_img = np.sum((np.power(mixedpure, 2)), 0) / np.power(10, snr / 10) / nRow / nCol
    noise_img = np.random.normal(0, np.sqrt(var_img), [nRow * nCol, nBand])

    mixed = np.transpose(mixedpure + noise_img)  # channels*(rows*columns)
    # self.data = mixed
    return mixed, var_img


if __name__ == '__main__':
    data_loc = "./DATA/dip_data/unmixing3/mixed.mat"
    gt_loc = "./DATA/dip_data/unmixing3/OtherInfo/A.mat"
    abf_true_loc = "./DATA/dip_data/unmixing3/OtherInfo/abf.mat"

    debug = True
    verbose = True
    num_endm = 4    # 混合影像数据的端元数
    SNR_noise = [10, 20, 30, 40]


    if debug:
        for s in range(0, len(SNR_noise)):
            g = SNR_noise[s]
            # file.write('## SNR = : %s \n\n' % g)  # SNR的大小
            mixed = load_data(data_loc)
            nRow = mixed.shape[0]
            nCol = mixed.shape[1]
            nBand = mixed.shape[2]
            data = addnoise(mixed, g)[0]
            data = np.reshape(data,[nRow,nCol,nBand])
            sio.savemat('mixed'+str(g),{'mixed':data})