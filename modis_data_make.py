import pdb
import tifffile as tiff
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
img_dir = '/mnt/Data/SU/data_results/MODIS-unmix/'
img_name = 'mosaic-1-5_all-channels.tiff'
label_name = 'CDL_2019_20210911071101_1142041687.tiff'
img = tiff.imread(img_dir+img_name)  #(256, 256, 23)
img_size = img.shape
label = tiff.imread(img_dir+label_name) #(2134, 2134)
##pdb.set_trace()

label_list = np.unique(label) # 38
k = len(label_list)
S_mat = np.zeros((img_size[0], img_size[1], k))
img_resize = np.resize(img,(2134,2134,23)) #(2134, 2134, 23)
A_mat = np.zeros((img_size[2], k))
for c in range(k):
    #pdb.set_trace()
    pos = np.where(label==label_list[c])
    A_mat[:,c] = np.mean(img_resize[pos[0],pos[1]],0)

#plt.plot(A_mat)
#plt.show()
#pdb.set_trace()
for row in range(img_size[0]):
    for col in range(img_size[1]):
        for c in range(k):
            ind_label = (np.int(row*8.33), np.int(col*8.33)) #top-left pixel
            #pdb.set_trace()
            patch_label = label[ind_label[0]:ind_label[0]+8, ind_label[1]:ind_label[1]+8]
            S_mat[row,col,c] = np.count_nonzero(patch_label==label_list[c])/(8*8)

#pdb.set_trace()
sio.savemat(img_dir+'raw_data/A.mat', {'A': A_mat})
sio.savemat(img_dir+'raw_data/S.mat', {'S': S_mat})
sio.savemat(img_dir+'raw_data/img.mat', {'img': img})

'''fig, axs = plt.subplots(38)
axs[0].imshow(S_mat[:,:,0])
axs[1].imshow(S_mat[:,:,1])
axs[2].imshow(S_mat[:,:,2])
axs[3].imshow(S_mat[:,:,3])
axs[4].imshow(S_mat[:,:,4])
axs[5].imshow(S_mat[:,:,5])
axs[6].imshow(S_mat[:,:,6])
axs[7].imshow(S_mat[:,:,7])
axs[8].imshow(S_mat[:,:,8])
axs[9].imshow(S_mat[:,:,9])
axs[10].imshow(S_mat[:,:,10])
axs[11].imshow(S_mat[:,:,11])
axs[12].imshow(S_mat[:,:,12])
axs[13].imshow(S_mat[:,:,13])
axs[14].imshow(S_mat[:,:,14])
axs[15].imshow(S_mat[:,:,15])
axs[16].imshow(S_mat[:,:,16])
axs[17].imshow(S_mat[:,:,17])
axs[18].imshow(S_mat[:,:,18])
axs[19].imshow(S_mat[:,:,19])
axs[20].imshow(S_mat[:,:,20])
axs[21].imshow(S_mat[:,:,21])
axs[22].imshow(S_mat[:,:,22])
axs[23].imshow(S_mat[:,:,23])
axs[24].imshow(S_mat[:,:,24])
axs[25].imshow(S_mat[:,:,25])
axs[26].imshow(S_mat[:,:,26])
axs[27].imshow(S_mat[:,:,27])
axs[28].imshow(S_mat[:,:,28])
axs[29].imshow(S_mat[:,:,29])
axs[30].imshow(S_mat[:,:,30])
axs[31].imshow(S_mat[:,:,31])
axs[32].imshow(S_mat[:,:,32])
axs[33].imshow(S_mat[:,:,33])
axs[34].imshow(S_mat[:,:,34])
axs[35].imshow(S_mat[:,:,35])
axs[36].imshow(S_mat[:,:,36])
axs[37].imshow(S_mat[:,:,37])
plt.show()'''

# 2 4 5 9 31 (-1) : more abundant
