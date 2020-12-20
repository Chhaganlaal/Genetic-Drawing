import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import string
import random

def convert_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def resize_image(img):
    height, width = img.shape

    resized_image = np.copy(img)

    if max(height, width)>1024:
        ratio = min(1024/height, 1024/width)
        new_h = int(height*ratio)
        new_w = int(width*ratio)

        resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return resized_image

def prep(img):
    img = convert_to_gray(img)
    img = resize_image(img)

    return img

def util_sample_from_img(img):
    #possible positions to sample
    pos = np.indices(dimensions=img.shape)
    pos = pos.reshape(2, pos.shape[1]*pos.shape[2])
    img_flat = np.clip(img.flatten() / img.flatten().sum(), 0.0, 1.0)
    return pos[:, np.random.choice(np.arange(pos.shape[1]), 1, p=img_flat)]

def draw(event, former_x, former_y, flags, param):

    if event==cv2.EVENT_LBUTTONDOWN:
        param['drawing']=True
        param['current_former_x'], param['current_former_y'] = former_x, former_y

    elif event==cv2.EVENT_MOUSEMOVE:
        if param['drawing']==True:
            cv2.line(param['im'], (param['current_former_x'], param['current_former_y']), (former_x, former_y), (255,255,255), param['brush_size'])
            cv2.line(param['mask'], (param['current_former_x'], param['current_former_y']), (former_x, former_y), (255,255,255), param['brush_size'])
            param['current_former_x'] = former_x
            param['current_former_y'] = former_y
            #print former_x,former_y
    elif event==cv2.EVENT_LBUTTONUP:
        param['drawing']=False
        cv2.line(param['im'], (param['current_former_x'], param['current_former_y']), (former_x, former_y), (255,255,255), param['brush_size'])
        cv2.line(param['mask'], (param['current_former_x'], param['current_former_y']), (former_x, former_y), (255,255,255), param['brush_size'])
        param['current_former_x'] = former_x
        param['current_former_y'] = former_y
    elif event==cv2.EVENT_MOUSEWHEEL:
        if flags<0:
            param['brush_size'] = min(param['brush_size']+5, 75)
        elif flags>0:
            param['brush_size'] = max(param['brush_size']-5, 5)

def create_custom_mask(img, blur):

    param = {
        'drawing': False,
        'current_former_x': -1,
        'current_former_y': -1,
        'im': img,
        'brush_size': 15
    }

    param['mask'] = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    cv2.namedWindow("Create Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Create Mask", resize_image(param['im']).shape[1], resize_image(param['im']).shape[0])
    cv2.setMouseCallback("Create Mask", draw, param)
    while(1):
        cv2.imshow("Create Mask", param['im'])
        if cv2.waitKey(1) & 0xff==13:
            break

    cv2.destroyWindow("Create Mask")
    param['mask'] = cv2.GaussianBlur(param['mask'], (0,0), blur, cv2.BORDER_DEFAULT)

    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("mask", resize_image(param['mask']).shape[1], resize_image(param['mask']).shape[0])
    cv2.imshow("mask", param['mask'])
    cv2.waitKey(0)
    cv2.destroyWindow("mask")

    return param['mask']
