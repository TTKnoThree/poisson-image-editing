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


def poisson_edit(source, target, mask, offset, kernel_size, spin=False, light=1.0):
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
    
    # add guassian filter to mask area
    if kernel_size:
        guassian_target = cv2.GaussianBlur(target, (kernel_size, kernel_size), 0)
        if spin:
            tmp1 = cv2.flip(guassian_target, 0)
            tmp2 = cv2.flip(guassian_target, 1)
            tmp3 = cv2.flip(guassian_target, -1)
            guassian_target = 0.25*guassian_target + 0.25*tmp1 + 0.25*tmp2 + 0.25*tmp3
            guassian_target *= light
    else:
        guassian_target = target.copy()
        
    alpha = mask.copy()/255.0
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

    mat_A = mat_A.tocsc()

    mask_flat = mask.flatten()    
    
    # target_gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    result = target.copy()
    for channel in range(source.shape[2]):
        source_flat = source[y_min:y_max, x_min:x_max, channel].flatten()
        target_flat = guassian_target[y_min:y_max, x_min:x_max, channel].flatten()
        # target_flat = target_gray[y_min:y_max, x_min:x_max].flatten()

        #concat = source_flat*mask_flat + target_flat*(1-mask_flat)
        
        # inside the mask:
        # \Delta f = div v = \Delta g       
        mat_b = laplacian.dot(source_flat)

        # outside the mask:
        # f = t
        mat_b[mask_flat==0] = target_flat[mask_flat==0]
        
        x = spsolve(mat_A, mat_b)
        x = x.reshape((y_range, x_range))
        x[x > 255] = 255
        x[x < 0] = 0
        x = x.astype('uint8')
        #x = cv2.normalize(x, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        #print(x.shape)

        result[y_min:y_max, x_min:x_max, channel] = x * alpha + target[y_min:y_max, x_min:x_max, channel] * (1-alpha)

    return result

def main():    
    scr_dir = 'figs/example'
    out_dir = scr_dir
    source = cv2.imread(path.join(scr_dir, "source3.jpg")) 
    target = cv2.imread("figs/background/target1.jpg")    
    mask = cv2.imread(path.join(scr_dir, "target_matte.png"), 
                      cv2.IMREAD_GRAYSCALE) 
    offset = (0,0)
    f = 0.5
    source = cv2.resize(source, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    target = cv2.resize(target, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    kernel_size = 5
    for i in range(5, 10):
        kernel_size = 5 + i * 50
        result = poisson_edit(source.copy(), target.copy(), mask.copy(), offset, kernel_size)
        cv2.imwrite(path.join(out_dir, f"possion_{kernel_size}.png"), result)
    

if __name__ == '__main__':
    main()
