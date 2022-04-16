# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 21:17:42 2022

@author: C846604
"""

import cv2 as cv
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, lfilter_zi
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

#%%
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    zi = lfilter_zi(b, a)
    y = lfilter(b, a, data, zi=data[0]*zi)[0]
    return y




def find_bent_edges(all_peaks_fltr, moving_steps):
    
    bent_edges = pd.DataFrame()

    for i_ in range(all_peaks_fltr['cluster_id'].max()+1):
        
        df_fltr = all_peaks_fltr[all_peaks_fltr['cluster_id'] == i_]

        # =======================        
        x = df_fltr['middle_index']
        y = df_fltr['peak_real']
        
        xq = np.arange(x.min(), x.max()+1)
        yq = np.interp(xq, x, y)
        
        # =======================
        # Filter requirements.
        order = 2
        fs = 1     # sample rate, Hz
        cutoff = 0.00004 * moving_steps  # desired cutoff frequency of the filter, Hz
        curv_filtered = butter_lowpass_filter(yq, cutoff, fs, order)
        
        # =======================
        z = np.polyfit(xq, curv_filtered, 1)
        p = np.poly1d(z)
        max_dev = np.abs(p(xq) - curv_filtered).max()
        
        if max_dev > 20:
            bent_edges = pd.concat([bent_edges, df_fltr])
    
    return bent_edges


def find_boundaries(img_gray_crp):
    
    sf = 200
    dst = cv.resize(img_gray_crp, (sf, sf))
    n = 11
    src = cv.GaussianBlur(dst, (n, n), 0)
    kernel_size = 3
    laplacian = cv.Laplacian(src,-1, ksize=kernel_size,scale=5, delta=0,  borderType=cv.BORDER_ISOLATED)
    abs_dst = cv.convertScaleAbs(laplacian)
    filter_output = np.sum(abs_dst, axis=1)
    filter_output = filter_output / filter_output.max()
    peaks, _ = find_peaks(filter_output, height=0.4, distance=20)
    peaks = np.asarray(peaks * img_gray_crp.shape[0] / sf).astype(int)
    
    return peaks


def find_all_peaks(img_gray, moving_steps, windows_size):
    
    all_peaks = pd.DataFrame()
    
    for ind0 in range(0, img_gray.shape[1], moving_steps):
        
        ind1 = ind0 + windows_size
        
        img_gray_crp = img_gray[:, ind0:ind1].copy()
#        img_gray_crp_raw = img_gray_raw[:, ind0:ind1].copy()
        
        peaks = find_boundaries(img_gray_crp) / img_gray_crp.shape[0]
    
        middle_index = int(ind0 + img_gray_crp.shape[1]/2)
        for peak in peaks:
            
            peak_real = int(peak * img_gray_crp.shape[0])
            
#            if it_is_a_top_edge(img_gray_crp_raw, peak_real):
#                peak_status = 't'
#            else:
#                peak_status = 'b'
            peak_status = 'unknown'
            
            all_peaks = all_peaks.append({
                    'middle_index': middle_index,
                    'peak_status': peak_status,
                    'peak_normalized': peak,
                    'peak_real': peak_real
                        }, ignore_index=True)
    
    return all_peaks

def brightness(img, minimum_brightness = 10):
    
    cols, rows = img.shape[:2]
    brightness = np.sum(img) / (255 * cols * rows)
    ratio = brightness / minimum_brightness
    
    bright_img = cv.convertScaleAbs(img, alpha = 1 / ratio, beta = 16)
    
    return bright_img



def filter_all_peaks_step_01(df, moving_steps):

    X = np.asarray([df["middle_index"],
                    df["peak_real"]*20]).transpose()
    
    clustering = DBSCAN(eps=moving_steps*10, min_samples=5).fit(X)
    
    all_peaks_fltr = pd.DataFrame()
    
    next_id = 0
    for i_ in range(clustering.labels_.max()+1):
        
        df_fltr = df[clustering.labels_ == i_]
        
        if len(df_fltr) < 10:
            continue
        
        if df_fltr['middle_index'].max() - df_fltr['middle_index'].min() < 1000:
            continue
        
        df_fltr['cluster_id'] = next_id
        
        all_peaks_fltr = pd.concat([all_peaks_fltr, df_fltr])
        
        next_id += 1
    
    all_peaks_fltr = all_peaks_fltr.reset_index()
    
    return all_peaks_fltr


def main(img_gray, bbox):
    
    # ==================
    windows_size = 300
    moving_steps = 50
    
    # ==================
    x0, y0, x1, y1 = bbox
    img_gray_raw = img_gray[y0:y1, x0:x1]
    img_gray = brightness(img_gray_raw, minimum_brightness = 0.5)
    
    # ==================
    all_peaks = find_all_peaks(img_gray, moving_steps, windows_size)
    
    all_peaks_fltr = filter_all_peaks_step_01(all_peaks, moving_steps)
    
    bent_edges = find_bent_edges(all_peaks_fltr, moving_steps)
    
    return bent_edges