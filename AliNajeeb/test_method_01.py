# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 21:21:33 2022

@author: C846604
"""
import cv2 as cv
import os
import numpy as np
from bent_finder_method_01 import main, brightness

#%%
def plot_each_edge(bent_edges_group_i, cluster_id, img_clr):
    
    a = np.array(bent_edges_group_i[['middle_index', 'peak_real']]).astype(np.int32)
    img_clr = cv.polylines(img_clr, 
              [a], 
              isClosed = False,
              color = (0,0,255),
              thickness = 3, 
              lineType = cv.LINE_4)
    
    img_clr = cv.putText(img_clr, f"{cluster_id}",
                         (int(bent_edges_group_i['middle_index'].mean()),
                          int(bent_edges_group_i['peak_real'].mean()+50)),
                         cv.FONT_HERSHEY_COMPLEX, 2, (255,255,255), 4)

    return img_clr


def save_it(dst_image, dst_path):
    
    dst_image = brightness(dst_image, minimum_brightness = 0.5)
    
    pto = os.path.join(r"C:\Users\C846604\Documents\bent_center_sill\sample_data\outputs",
                       os.path.basename(dst_path))
    return cv.imwrite(pto, dst_image)

#%%
# bounding box format
# bbox = (x0, y0, x1, y1)
image_metadata = {
    '1':
        {'pti': r"C:\Users\C846604\Documents\bent_center_sill\sample_data\Defect 1.jpg",
         'bbox': (3625, 1024, 17925, 2048)},
    '2':
        {'pti': r"C:\Users\C846604\Documents\bent_center_sill\sample_data\Defect 1.jpg",
         'bbox': (3625, 0, 17925, 1024)},
    '3':
        {'pti': r"C:\Users\C846604\Documents\bent_center_sill\sample_data\Defect 2.jpg",
         'bbox': (3625, 0, 17925, 1024)},
    '4':
        {'pti': r"C:\Users\C846604\Documents\bent_center_sill\sample_data\Straight Center Sill 5.jpg",
         'bbox': (3625, 0, 17925, 1024)},
    '5':
        {'pti': r"C:\Users\C846604\Documents\bent_center_sill\sample_data\Straight Center Sill 4.jpg",
         'bbox': (3625, 0, 17925, 1024)},
    '6':
        {'pti': r"C:\Users\C846604\Documents\bent_center_sill\Sample Images\Bent (Defect)\KHW2L1BNUWLKPOLAQOHZ94AEKLZ3.JPEG",
         'bbox': (2683, 0, 18450, 2048)},
    }

if __name__ == "__main__":

    image_id = '3'
    pti = image_metadata[image_id]['pti']
    bbox = image_metadata[image_id]['bbox']
    
    # ================
    img_gray = cv.imread(pti, 0)
    x0, y0, x1, y1 = bbox
    
    # ================
    bent_edges = main(img_gray, bbox)
    
    # ================
    if len(bent_edges) > 0:
        img_clr = cv.imread(pti)
        img_clr = img_clr[y0:y1, x0:x1, :]
        for cluster_id, group in bent_edges.groupby("cluster_id"):
            img_clr = plot_each_edge(group, cluster_id, img_clr)
    save_it(img_clr, pti)