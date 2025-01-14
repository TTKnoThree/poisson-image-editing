import cv2
import numpy as np

from paint_mask import MaskPainter
from move_mask import MaskMover
from poisson_image_editing import poisson_edit
from advance_poisson_image_editing import advance_poisson_edit
from mask_separation.detect_mask import detect_mask
import MODNet.inference as modnet

import argparse
from os import path

parser = argparse.ArgumentParser(description='Add arguments for main.py')
parser.add_argument('-s', '--source', type=str, required=True, help='The path of source image.')
parser.add_argument('-t', '--target', type=str, required=True, help='The path of target image.')
parser.add_argument('-p', '--pattern', type=str, default=None, help='The path of mask pattern, if = None, don\'t change pattern.')
parser.add_argument('-r', '--resize', type=float, default=1.0, help='resize the target image with this ratio.')
parser.add_argument('-g', '--gaussian_kernel', type=int, default=555, help='The size of gaussian kernel.')
parser.add_argument('-l', '--light_ratio', type=float, default=None, help='The light ratio of poisson process.')
parser.add_argument('-a', '--automatic', action='store_true', help='Use MODnet for automatic matting.')
parser.add_argument('-d', '--direct', action='store_true', help='Use direct concat instead of poisson image editing.')

args = parser.parse_args()

if __name__ == '__main__':
    # 1. get&resize source, target, pattern images
    source = cv2.imread(args.source)
    target = cv2.imread(args.target)

    if source is None or target is None:
        print('Source or target image not exist.')
        exit()
        
    if args.pattern:
        pattern = cv2.imread(args.pattern)
        if pattern is None:
            print('Pattern image not exist.')
            exit()
    else:
        if args.resize != 1.0:
            target = cv2.resize(target, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)
        if source.shape[0] > target.shape[0] or source.shape[1] > target.shape[1]:
            fx = target.shape[0] / source.shape[0]
            fy = target.shape[1] / source.shape[1]
            f = min(fx, fy)
            source = cv2.resize(source, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

    # 2. get foreground and target_mask from source
    # draw target mask
    if args.automatic:
        print('MODNet is generating source mask...')
        target_mask = modnet.inference(source)
    else:
        print('Please highlight the object to use for poisson-image-editing.\n')
        mp = MaskPainter(path.dirname(args.source), source)
        target_mask = mp.paint_mask() 
        target_mask = target_mask.astype(np.uint8)   
    cv2.imwrite('matting.jpg', target_mask)
    foreground = (target_mask/255*source).astype(np.uint8)

    if args.pattern:
        # 3. detect face mask from foreground
        print('mask_seperation is generating pattern mask...')
        pattern_mask = detect_mask(foreground)
        if pattern_mask is None:
            print('Could not detect face mask automaticly, please highlight the face mask manually.\n')
            mp = MaskPainter(path.dirname(args.source), source)
            pattern_mask = mp.paint_mask() 
            # resize target and source
        cv2.imwrite('mask.jpg', pattern_mask)
        if args.resize != 1.0:
            target = cv2.resize(target, None, fx=args.resize, fy=args.resize, interpolation=cv2.INTER_AREA)
        if source.shape[0] > target.shape[0] or source.shape[1] > target.shape[1]:
            fx = target.shape[0] / source.shape[0]
            fy = target.shape[1] / source.shape[1]
            f = min(fx, fy)
            source = cv2.resize(source, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            target_mask = cv2.resize(target_mask, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            pattern_mask = cv2.resize(pattern_mask, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        # resize pattern 
        if pattern.shape[0] < source.shape[0] or pattern.shape[1] < source.shape[1]:
            fx = source.shape[0] / pattern.shape[0]
            fy = source.shape[1] / pattern.shape[1]
            f = max(fx, fy)
            pattern = cv2.resize(pattern, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)

        pattern_mask = pattern_mask.astype(np.uint8)

        # 4. pick pattern for face mask
        print('Please move the face mask to pick pattern.\n')
        mm = MaskMover(pattern, pattern_mask)
        M, _ = mm.move_mask()
        
        ## modify by cq, perform affine transformation on source before poisson editing
        pattern = cv2.warpAffine(pattern, M, (source.shape[1], source.shape[0]), flags=cv2.WARP_INVERSE_MAP) 

        # 5. blend picked pattern with source
        pattern_mask = cv2.cvtColor(pattern_mask, cv2.COLOR_RGB2GRAY) 
        ## pattern_mask didn't need to be transformed
        offset = 0, 0
        source = advance_poisson_edit(pattern, source, pattern_mask, offset) 
        cv2.imwrite('tmp.jpg', source)
        # exit()
        print(f'pattern -> source: pattern.shape={pattern.shape}, pattern_mask.shape={pattern_mask.shape}, source.shape={source.shape}') 
        
    # 6. adjust person position for target image
    print('Please move the object to desired location to apparate.\n')
    mm = MaskMover(target, target_mask)
    M, target_mask = mm.move_mask()            

    ## modify by cq, perform affine transformation on source before poisson editing
    source = cv2.warpAffine(source, M, (target.shape[1], target.shape[0]))

    # blend
    print('Blending ...')
    target_mask = target_mask.astype(np.uint8)
    target_mask = cv2.cvtColor(target_mask, cv2.COLOR_RGB2GRAY) 
    
    ## as affine transformation is already performed on source, the offset should be set to 0 to avoid repeated transformation
    offset = 0, 0

    if args.direct:
        target_mask = target_mask[:, :, np.newaxis]
        target_mask = np.concatenate([target_mask, target_mask, target_mask], axis=2)
        poisson_blend_result = np.where(target_mask == 0, target, source)
    else:
        poisson_blend_result = poisson_edit(source.copy(), target.copy(), target_mask.copy(), offset, args.gaussian_kernel, spin=True, light=args.light_ratio)


    print(f'source -> target: source.shape={source.shape}, target_mask.shape={target_mask.shape}, poisson_blend_result.shape={poisson_blend_result.shape}')
    
    # if args.resize != 1.0:
    #     target = cv2.resize(target, None, fx=1/args.resize, fy=1/args.resize, interpolation=cv2.INTER_AREA)
    
    cv2.imwrite(path.join(path.dirname(args.target), 'result.png'), 
                poisson_blend_result)

    # for i in [55, 255, 555, 1055]:
    #     if args.direct:
    #         target_mask = target_mask[:, :, np.newaxis]
    #         target_mask = np.concatenate([target_mask, target_mask, target_mask], axis=2)
    #         poisson_blend_result = np.where(target_mask == 0, target, source)
    #     else:
    #         poisson_blend_result, guassian_target = poisson_edit(source.copy(), target.copy(), target_mask.copy(), offset, i, spin=True)
    
    
    #     print(f'source -> target: source.shape={source.shape}, target_mask.shape={target_mask.shape}, poisson_blend_result.shape={poisson_blend_result.shape}')
        
    #     # if args.resize != 1.0:
    #     #     target = cv2.resize(target, None, fx=1/args.resize, fy=1/args.resize, interpolation=cv2.INTER_AREA)
        
    #     cv2.imwrite(path.join(path.dirname(args.source), f'result_spin_{i}.png'), 
    #                 poisson_blend_result)
    #     cv2.imwrite(path.join(path.dirname(args.source), f'gaussian_result_spin_{i}.png'), 
    #                 guassian_target)

    
    
    print('Done.\n')
