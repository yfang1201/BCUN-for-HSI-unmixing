import os
import scipy.io as sio
import scipy.optimize.nnls
import scipy
import numpy as np
from matplotlib import pyplot as plt

from tempfile import TemporaryFile

import pdb
indian_pine_loc = "./DATA/Indian_pines_corrected.mat"
indian_pine = sio.loadmat(indian_pine_loc)['indian_pines_corrected']

indian_pine_gt_loc = "./DATA/Indian_pines_gt.mat"
'''indian_pine_gt = sio.loadmat(indian_pine_gt_loc)['indian_pines_gt']
print(np.shape(indian_pine_gt))
print(np.shape(indian_pine))
#pdb.set_trace()

plt.imshow(indian_pine_gt)
plt.show()


plt.imshow(indian_pine_gt[100:113,27:45])
plt.show()
'''
ROI_indian_pine = np.reshape(indian_pine[100:113,27:45,:],(234,200))

n = ROI_indian_pine.shape[0]
p = ROI_indian_pine.shape[1]

var_noi = np.var(ROI_indian_pine, axis = 0, keepdims = True).squeeze()
tmp = np.sum(pow(ROI_indian_pine, 2), 0)/(var_noi*n)
snr_noi = 10*np.log10(tmp)
snr_noi[snr_noi < 0] = 0
cen_snr_noi = (snr_noi - np.mean(snr_noi))  ## this is the centered snr
nor_cen_snr_noi = cen_snr_noi/np.max(cen_snr_noi) ## this is the centered and normalized snr
scal_nor_cen_snr_noi = 5*nor_cen_snr_noi

'''print(np.shape(ROI_indian_pine))
print(np.shape(var_noi))
print(var_noi)'''
plt.plot(snr_noi)
plt.show()
plt.plot(cen_snr_noi)
plt.show()
plt.plot(nor_cen_snr_noi)
plt.show()
plt.plot(scal_nor_cen_snr_noi)
plt.show()
## save and load the variable of norm_var_noi//// but it does not work well...
'''var_file = TemporaryFile()
np.save(var_file, norm_var_noi)


_ = var_file.seek(0) # Only needed here to simulate closing & reopening file
np.load(var_file)'''
