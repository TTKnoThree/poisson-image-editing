import cv2
import numpy as np

from paint_mask import MaskPainter
from move_mask import MaskMover
from poisson_image_editing import poisson_edit
import MODNet.inference as modnet

import argparse
from os import path

parser = argparse.ArgumentParser(description='Add arguments for main.py')
parser.add_argument('-a', '--automatic', action='store_true', help='Use MODnet for automatic matting.')
parser.add_argument('-s', '--source', type=str, required=True, help='The path of source image.')
parser.add_argument('-t', '--target', type=str, required=True, help='The path of target image.')
parser.add_argument('-m', '--mask', type=str, default=None, help='The path of existed mask image.')
parser.add_argument('-r', '--resize', type=float, default=1.0, help='resize the target image with this ratio')

args = parser.parse_args()

if __name__ == '__main__':
    source = cv2.imread(args.source)
    target = cv2.imread(args.target)
    
    if source is None or target is None:
        print('Source or target image not exist.')
        exit()

    # resize images size
    if args.resize != 1.0:
        target = cv2.resize(target, None, fx=args.resize, fy=args.resize)
    if source.shape[0] > target.shape[0] or source.shape[1] > target.shape[1]:
        # fx = source.shape[0] / target.shape[0]
        # fy = source.shape[1] / target.shape[1]
        # f = max(fx, fy)
        # target = cv2.resize(target, None, fx=f, fy=f)
        fx = target.shape[0] / source.shape[0]
        fy = target.shape[1] / source.shape[1]
        f = min(fx, fy)
        source = cv2.resize(source, None, fx=f, fy=f)

    # draw the mask
    if args.automatic:
        print('MODNet is generating mask...')
        mask = modnet.inference(source)
        cv2.imwrite(path.join(path.dirname(args.source), 'target_matte.png'), 
            mask)
    elif not args.mask:
        print('Please highlight the object to disapparate.\n')
        mp = MaskPainter(path.dirname(args.source), source)
        mask = mp.paint_mask() 
        mask = mask.astype(np.uint8)
    else:
        mask = cv2.imread(args.mask)   
    
    if mask.shape[0] > target.shape[0] or mask.shape[1] > target.shape[1]:
        fx = target.shape[0] / mask.shape[0]
        fy = target.shape[1] / mask.shape[1]
        f = min(fx, fy)
        mask = cv2.resize(mask, None, fx=f, fy=f)
    
    # adjust mask position for target image
    print('Please move the object to desired location to apparate.\n')
    mm = MaskMover(target, mask)
    offset_x, offset_y, target_mask = mm.move_mask()            

    # blend
    print('Blending ...')
    target_mask = target_mask.astype(np.uint8)
    target_mask = cv2.cvtColor(target_mask, cv2.COLOR_RGB2GRAY) 
    offset = offset_x, offset_y

    poisson_blend_result = poisson_edit(source, target, target_mask, offset)
    
    cv2.imwrite(path.join(path.dirname(args.source), 'target_result.png'), 
                poisson_blend_result)
    
    
    print('Done.\n')
