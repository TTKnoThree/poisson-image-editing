import argparse
import numpy as np
import cv2
from os import path
 

class MaskMover():
    def __init__(self, image, mask):
        # initialize mask size
        if image.shape[0] != mask.shape[0]:
            block = np.zeros((image.shape[0]-mask.shape[0], mask.shape[1], 3))
            mask = np.concatenate((mask, block), axis=0)
        if image.shape[1] != mask.shape[1]:
            block = np.zeros((mask.shape[0], image.shape[1]-mask.shape[1], 3))
            mask = np.concatenate((mask, block), axis=1)
        
        self.image = image
        self.original_mask = mask
        self.image_copy = self.image.copy()

        # self.original_mask_copy = np.zeros(self.original_mask.shape)
        # self.original_mask_copy[np.where(self.original_mask!=0)] = 255

        self.mask = self.original_mask.copy()

        self.to_move = False
        self.shift_x = 0
        self.shift_y = 0
        self.xi = 0
        self.yi = 0
        self.is_first = True
        self.resize = 100
        
        self.window_name = "Move the mask. s:save; r:reset; q:quit"


    def _blend(self, image, mask):
        ret = image.copy()
        alpha = 0.3    
        ret[mask != 0] = ret[mask != 0]*alpha + 255*(1-alpha)
        return ret.astype(np.uint8)


    def _move_mask_handler(self, event, x, y, flags, param): 
        # 鼠标左键按下   
        if event == cv2.EVENT_LBUTTONDOWN:  
            self.to_move = True
            self.xi, self.yi = x, y

        # 鼠标移动
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.to_move:
                self.shift_x += (x-self.xi)*100.0/self.resize
                self.shift_y += (y-self.yi)*100.0/self.resize
                self.M = np.float32([[self.resize/100.0,0,self.resize/100.0*(self.shift_x)],
                                [0,self.resize/100.0,self.resize/100.0*(self.shift_y)]])
                self.mask = cv2.warpAffine(self.original_mask,self.M,
                                      (self.image.shape[1],
                                       self.image.shape[0]))
                cv2.imshow(self.window_name, 
                           self._blend(self.image, self.mask))
                self.xi, self.yi = x, y

        # 鼠标左键释放
        elif event == cv2.EVENT_LBUTTONUP:
            self.to_move = False  


    def move_mask(self):
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar('mask_size', self.window_name, 100, 100, lambda x: None)
        cv2.setMouseCallback(self.window_name, 
                             self._move_mask_handler)
        self.resize = cv2.getTrackbarPos('mask_size', self.window_name)
        self.M = np.float32([[self.resize/100.0,0,0],[0,self.resize/100.0,0]])
        while True:  
            self.resize = cv2.getTrackbarPos('mask_size', self.window_name)
            rate = self.resize/100.0/self.M[0][0]
            self.M = self.M*rate
            self.mask = cv2.warpAffine(self.original_mask,self.M,
                                    (self.image.shape[1],
                                    self.image.shape[0]))                      
            cv2.imshow(self.window_name, 
                       self._blend(self.image, self.mask))
            key = cv2.waitKey(1) & 0xFF
 
            if key == ord("r"):
                self.image = self.image_copy.copy()
                self.mask = self.original_mask.copy()
                self.M = np.float32([[1.0,0,0],[0,1,0]])
                self.shift_x = 0
                self.shift_y = 0
                cv2.setTrackbarPos('mask_size', self.window_name, 100)
     
            elif key == ord("s"):
                break

            elif key == ord("q"):
                cv2.destroyAllWindows()
                exit()

        roi = self.mask
        cv2.imshow("Press any key to save the mask", roi)
        cv2.waitKey(0)

        # close all open windows
        cv2.destroyAllWindows()
        return self.M, self.mask


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-m", "--mask", required=True, help="Path to the mask")
    args = vars(ap.parse_args())
 
    mm = MaskMover(args["image"], args["mask"])
    offset_x, offset_y, _ = mm.move_mask()
    print(offset_x, offset_y)
    