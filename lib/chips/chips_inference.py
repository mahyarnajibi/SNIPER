#------------------------------------------------------------------
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Inference Chip Generation Module for AutoFocus
# by Mahyar Najibi and Bharat Singh
# -----------------------------------------------------------------
import os
import cv2
import math
import numpy as np


def gmask(mask, d, thresh_value=0.5, ms=16, im_width=0, im_height=0, cscale=1):           
    mask = mask.copy()
    iw =    int(math.ceil(float(im_width)/16))
    ih =    int(math.ceil(float(im_height)/16))
    kernel = np.ones((d,d),np.uint8)
    mask[np.where(mask>=thresh_value)] = 1
    mask[np.where(mask<thresh_value)] = 0
    mask = cv2.dilate(mask, kernel)
    mask *= 255

    _, cnts, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    chips = []
    nchips = -1
            
    while nchips != len(chips):
        nchips = len(chips)
        chips = []
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            cx = (x+x+w)/2
            cy = (y+y+h)/2
            w = max(ms, w)
            h = max(ms, h)

            if (cx+w/2 >= iw):
                x = iw - w if iw-w>=0 else 0
            elif (cx-w/2 < 0):
                x = 0
            else:
                x = cx - w/2

            if (cy+h/2 >= ih):
                y = ih - h if ih-h>=0 else 0
            elif (cy-h/2 < 0):
                y = 0
            else:
                y = cy - h/2
            mask[y:y+h,x:x+w] = 255
        _, cnts, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
        for cnt in cnts:
            x,y,w,h = cv2.boundingRect(cnt)
            cx = (x+x+w)/2
            cy = (y+y+h)/2
            w = max(ms, w)
            h = max(ms, h)

            if (cx+w/2 >= iw):
                x = iw - w if iw-w>=0 else 0
            elif (cx-w/2 < 0):
                x = 0
            else:
                x = cx - w/2

            if (cy+h/2 >= ih):
                y = ih - h if ih-h>=0 else 0
            elif (cy-h/2 < 0):
                y = 0
            else:
                y = cy - h/2

            chips.append([x,y,x+w,y+h])
    
    schips = []
    for c in chips:
        x1 = c[0] * 16
        y1 = c[1] * 16
        x2 = c[2] * 16
        y2 = c[3] * 16

        if x2 > im_width:
            x2 = im_width
            x1 = max(min(x1, x2 - ms*16), 0)
        if y2 > im_height:
            y2 = im_height
            y1 = max(min(y1, y2 - ms*16), 0)
        schips.append([x1/cscale, y1/cscale, x2/cscale, y2/cscale])

    return schips

def add_chips(roidb, maps, scale_id, cfg):
    def check_valid(det, chip, im_width, im_height):
        dx1 = det[0]
        dy1 = det[1]
        dx2 = det[2]
        dy2 = det[3]

        cx1 = chip[0]
        cy1 = chip[1]
        cx2 = chip[2]
        cy2 = chip[3]

        flag = True

        if dx1 == cx1:
            if cx1 != 0:
                flag = False
                return flag

        if dy1 == cy1:
            if cy1 != 0:
                flag = False
                return flag

        if dx2 == cx2:
            if dx2 < im_width:
                flag = False
                return flag

        if dy2 == cy2:
            if dy2 < im_height:
                flag = False
                return flag

        return flag
      
    min_target_size = cfg.TEST.SCALES[scale_id][0]
    max_target_size = cfg.TEST.SCALES[scale_id][1]
    total_area = 0
    chip_area = 0
    cropidc = 0
    for i, r in enumerate(roidb):
        cur_chips = []
        im_width = r['width']
        im_height = r['height']
        im_size_min = min(im_width, im_height)
        im_size_max = max(im_width, im_height)

        # Compute the scales for map and next scale
        cscale = float(min_target_size) / float(im_size_min)
        if np.round(cscale * im_size_max) > max_target_size:
            cscale = float(max_target_size) / float(im_size_max)

        tcscale = float(cfg.TEST.SCALES[scale_id+1][0]) / float(im_size_min)
        if np.round(tcscale * im_size_max) > cfg.TEST.SCALES[scale_id+1][1]:
            tcscale = float(cfg.TEST.SCALES[scale_id+1][1]) / float(im_size_max)

        total_area = total_area + (im_width * im_height * tcscale * tcscale)/(1000.*1000.)

        for j in range(len(maps[i])):
            cmap = maps[i][j][1]
            cur_crop = r['inference_crops'][j]
            crop_width = cur_crop[2] - cur_crop[0]
            crop_height = cur_crop[3] - cur_crop[1]
            chips = gmask(cmap, cfg.TEST.CHIP_HYPERPARAMS[scale_id][0], cfg.TEST.CHIP_HYPERPARAMS[scale_id][1],
                ms=cfg.TEST.CHIP_HYPERPARAMS[scale_id][2], im_width=crop_width*cscale, im_height=crop_height*cscale,
                cscale=cscale)
            for c in chips:
                c[0] += cur_crop[0]
                c[1] += cur_crop[1]
                c[2] += cur_crop[0]
                c[3] += cur_crop[1]

            for c in chips:
                tarea = (c[2] - c[0]) * (c[3] - c[1]) * tcscale * tcscale
                chip_area = chip_area + (tarea)/(1000.*1000.)
            cur_chips += chips

        roidb[i]['inference_crops'] = np.array(cur_chips)
    
    speed_up = 100.*chip_area / total_area
    print('Percent of pixels to be processed: {}'.format(speed_up))
    return [chip_area, total_area]