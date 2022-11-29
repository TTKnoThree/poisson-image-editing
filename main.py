import cv2
import numpy as np

from paint_mask import MaskPainter
from move_mask import MaskMover
from poisson_image_editing import poisson_edit
import MODNet.inference as modnet

#import argparse
import getopt
import sys
from os import path


def usage():
    print("Usage: python main.py [options] \n\n\
    Options: \n\
    \t-h\tPrint a brief help message and exits..\n\
    \t-a\t(Optional) Use MODnet for automatic matting.\n\
    \t-s\t(Required) Specify a source image.\n\
    \t-t\t(Required) Specify a target image.\n\
    \t-m\t(Optional) Specify a mask image with the object in white and other part in black, ignore this option if you plan to draw it later.")


if __name__ == '__main__':
    # parse command line arguments
    args = {}
    
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "h:a:s:t:m:p:")
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        print("See help: main.py -h")
        exit(2)
    for o, a in opts:
        if o in ("-h"):
            usage()
            exit()
        elif o in ("-a"):
            args["modnet"] = True
        elif o in ("-s"):
            args["source"] = a
        elif o in ("-t"):
            args["target"] = a
        elif o in ("-m"):
            args["mask"] = a        
        else:
            assert False, "unhandled option"
    
    #     
    if ("source" not in args) or ("target" not in args):
        usage()
        exit()
    
    #    
    source = cv2.imread(args["source"])
    target = cv2.imread(args["target"])
    
    if source is None or target is None:
        print('Source or target image not exist.')
        exit()

    # resize images size
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
    if "modnet" in args:
        print('MODNet is generating mask...')
        mask = modnet.inference(source)
    elif "mask" not in args:
        print('Please highlight the object to disapparate.\n')
        mp = MaskPainter(args["source"])
        mask = mp.paint_mask() 
    else:
        mask = cv2.imread(args["mask"])
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
    
    cv2.imwrite(path.join(path.dirname(args["source"]), 'target_result.png'), 
                poisson_blend_result)
    
    print('Done.\n')
