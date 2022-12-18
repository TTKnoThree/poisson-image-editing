import cv2
import numpy as np

from paint_mask import MaskPainter
from move_mask import MaskMover
from poisson_image_editing import poisson_edit
from mask_separation.detect_mask import detect_mask
import MODNet.inference as modnet

import argparse
from os import path

parser = argparse.ArgumentParser(description='Add arguments for main.py')
parser.add_argument('-s', '--source', type=str, required=True, help='The path of source image.')
parser.add_argument('-t', '--target', type=str, required=True, help='The path of target image.')
parser.add_argument('-p', '--pattern', type=str, required=True, help='The path of mask pattern.')
parser.add_argument('-r', '--resize', type=float, default=1.0, help='resize the target image with this ratio')
parser.add_argument('-a', '--automatic', action='store_true', help='Use MODnet for automatic matting.')

args = parser.parse_args()

if __name__ == '__main__':
    # 1. get source, target, pattern images
    source = cv2.imread(args.source)
    target = cv2.imread(args.target)
    pattern = cv2.imread(args.pattern)
    
    if source is None or target is None or pattern is None:
        print('Source or target or pattern image not exist.')
        exit()

    # resize images size
    if args.resize != 1.0:
        target = cv2.resize(target, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)
    if source.shape[0] > target.shape[0] or source.shape[1] > target.shape[1]:
        fx = target.shape[0] / source.shape[0]
        fy = target.shape[1] / source.shape[1]
        f = min(fx, fy)
        source = cv2.resize(source, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    if pattern.shape[0] > source.shape[0] or pattern.shape[1] > source.shape[1]:
        fx = source.shape[0] / pattern.shape[0]
        fy = source.shape[1] / pattern.shape[1]
        f = min(fx, fy)
        pattern = cv2.resize(pattern, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    # if pattern.shape[0] < source.shape[0] or pattern.shape[1] < source.shape[1]:
    #     fx = source.shape[0] / pattern.shape[0]
    #     fy = source.shape[1] / pattern.shape[1]
    #     f = max(fx, fy)
    #     pattern = cv2.resize(pattern, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


    # 2. pattern -> source
    # draw pattern mask
    if args.automatic:
        print('mask_seperation is generating pattern mask...')
        pattern_mask = detect_mask(source)
    else:
        print('Please highlight the object to disapparate.\n')
        mp = MaskPainter(path.dirname(args.source), source)
        pattern_mask = mp.paint_mask() 
        pattern_mask = pattern_mask.astype(np.uint8)
        
    # adjust mask position for target image
    print('Please move the object to desired location to apparate.\n')
    mm = MaskMover(pattern, pattern_mask)
    offset_x, offset_y, pattern_mask = mm.move_mask()
    print(f'2. {offset_x}, {offset_y}, {pattern_mask.shape}')

    # blend
    print('Blending ...')
    pattern_mask = pattern_mask.astype(np.uint8)
    pattern_mask = cv2.cvtColor(pattern_mask, cv2.COLOR_RGB2GRAY) 
    offset = offset_x, offset_y

    source = poisson_edit(pattern, source, pattern_mask, offset, reverse=True)
    
    print(f'2. pattern -> source: {pattern.shape}, {pattern_mask.shape}, {source.shape}')


    # 3. source -> target

    # draw target mask
    if args.automatic:
        print('MODNet is generating source mask...')
        target_mask = modnet.inference(source)
        # cv2.imwrite(path.join(path.dirname(args.source), 'source_mask.png'), 
        #     mask)
    else:
        print('Please highlight the object to disapparate.\n')
        mp = MaskPainter(path.dirname(args.source), source)
        target_mask = mp.paint_mask() 
        target_mask = target_mask.astype(np.uint8)
    
    # adjust mask position for target image
    print('Please move the object to desired location to apparate.\n')
    mm = MaskMover(target, target_mask)
    offset_x, offset_y, target_mask = mm.move_mask()
    print(f'3. {offset_x}, {offset_y}, {target_mask.shape}')

    # blend
    print('Blending ...')
    target_mask = target_mask.astype(np.uint8)
    target_mask = cv2.cvtColor(target_mask, cv2.COLOR_RGB2GRAY) 
    offset = offset_x, offset_y

    poisson_blend_result = poisson_edit(source, target, target_mask, offset)
    
    
    print(f'3. source -> target: {source.shape}, {target_mask.shape}, {poisson_blend_result.shape}')
    
    if args.resize != 1.0:
        target = cv2.resize(target, None, fx=1/args.resize, fy=1/args.resize, interpolation=cv2.INTER_AREA)
    
    cv2.imwrite(path.join(path.dirname(args.source), 'target_result.png'), 
                poisson_blend_result)

    
    
    print('Done.\n')
