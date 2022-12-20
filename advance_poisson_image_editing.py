"""Poisson image editing.

"""

import numpy as np
import cv2
import scipy.sparse
from scipy.sparse.linalg import spsolve

from os import path

def laplacian_matrix(n, m):
    """Generate the Poisson matrix. 

    Refer to: 
    https://en.wikipedia.org/wiki/Discrete_Poisson_equation

    Note: it's the transpose of the wiki's matrix 
    """
    mat_D = scipy.sparse.lil_matrix((m, m))
    mat_D.setdiag(-1, -1)
    mat_D.setdiag(4)
    mat_D.setdiag(-1, 1)
        
    mat_A = scipy.sparse.block_diag([mat_D] * n).tolil()
    
    mat_A.setdiag(-1, 1*m)
    mat_A.setdiag(-1, -1*m)
    
    return mat_A


def advance_poisson_edit(source, target, mask, offset):
    """The poisson blending function. 

    Refer to: 
    Perez et. al., "Poisson Image Editing", 2003.
    """

    # Assume: 
    # target is not smaller than source.
    # shape of mask is same as shape of target.
    y_max, x_max = target.shape[:-1]
    y_min, x_min = 0, 0

    x_range = x_max - x_min
    y_range = y_max - y_min
        
    M = np.float32([[1,0,offset[0]],[0,1,offset[1]]])
    source = cv2.warpAffine(source,M,(x_range,y_range))
    # noise = np.random.randint(0, 10, size=(x_range, y_range, 3))   
    # source += noise.astype(np.uint8) 
    # source[source>255] = 255
        
    mask = mask[y_min:y_max, x_min:x_max]    
    mask[mask != 0] = 1
    #mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    
    mat_A = laplacian_matrix(y_range, x_range)

    # for \Delta g
    laplacian = mat_A.tocsc()

    # set the region outside the mask to identity    
    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):
            if mask[y, x] == 0:
                k = x + y * x_range
                mat_A[k, k] = 1
                mat_A[k, k + 1] = 0
                mat_A[k, k - 1] = 0
                mat_A[k, k + x_range] = 0
                mat_A[k, k - x_range] = 0

    # corners
    # mask[0, 0]
    # mask[0, y_range-1]
    # mask[x_range-1, 0]
    # mask[x_range-1, y_range-1]

    # mat_A = mat_A.tocsc()
    cnt = 0
    total = (mask>0).sum()
    mask_flat = mask.flatten()  
    
    ##########################################################################
    # label the extra pixels in channel 0
    channel = 0
    label = np.zeros((y_range, x_range))
    mat_A_tmp = mat_A.copy()
    source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
    target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()       
    
    # inside the mask:
    # \Delta f = div v = \Delta g       
    alpha = 1
    mat_source = laplacian.dot(source_flat)*alpha
    mat_target = laplacian.dot(target_flat)*alpha
    mat_b = mat_source
    for y in range(1, y_range - 1):
        for x in range(1, x_range - 1):       
            if 10000*np.abs(mat_source[y*x_range+x])<np.abs(mat_target[y*x_range+x]) and mask[y, x] != 0:
                label[y,x] = 1
                cnt += 1
    ############################################################            
                    
    for channel in range(source.shape[2]):
        mat_A_tmp = mat_A.copy()
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = target[y_min:y_max, x_min:x_max, channel].flatten()       
        
        # inside the mask:
        # \Delta f = div v = \Delta g       
        alpha = 1
        mat_source = laplacian.dot(source_flat)*alpha
        mat_target = laplacian.dot(target_flat)*alpha
        mat_b = mat_source
        for y in range(1, y_range - 1):
            for x in range(1, x_range - 1):
                ############################################################
                if label[y, x]!=0:
                # 使用if False替换上面条件,即为改动前的版本
                # if False:
                    # source梯度较小，不改变该点像素值
                    k = x + y * x_range # flatten 之后的 index
                    mat_A_tmp[k, k] = 1
                    mat_A_tmp[k, k + 1] = 0 # flatten 前的左右
                    mat_A_tmp[k, k - 1] = 0
                    mat_A_tmp[k, k + x_range] = 0 # flatten 前的上下
                    mat_A_tmp[k, k - x_range] = 0
                    
                    mask_flat[k] = 0
                # if (y*x_range+x)%10000==0:
                #     print(y*x_range+x)
        mat_A_tmp = mat_A_tmp.tocsc()

        # outside the mask:
        # f = t
        mat_b[mask_flat==0] = target_flat[mask_flat==0]
        
        x = spsolve(mat_A_tmp, mat_b)
        #print(x.shape)
        
        x = x.reshape((y_range, x_range))
        # print(x[:30,:30])
        #print(x.shape)
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        #x = cv2.normalize(x, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        #print(x.shape)

        target[y_min:y_max, x_min:x_max, channel] = x
    print("保留原图像素点比例：", cnt/total)
    return target

def main():    
    scr_dir = 'figs/example'
    out_dir = scr_dir
    source = cv2.imread(path.join(scr_dir, "source1.jpg")) 
    target = cv2.imread(path.join(scr_dir, "target1.jpg"))    
    mask = cv2.imread(path.join(scr_dir, "mask1.png"), 
                      cv2.IMREAD_GRAYSCALE) 
    offset = (0,66)
    result = advance_poisson_edit(source, target, mask, offset)

    cv2.imwrite(path.join(out_dir, "possion1.png"), result)
    

if __name__ == '__main__':
    main()