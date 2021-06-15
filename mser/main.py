from preprocessing.data_loader import load_data
from preprocessing.data_preprocessing import apply_brightness_contrast
import os
import numpy as np
import time
import tifffile as tiff
import cv2
from glob import glob
from shapely.geometry import Polygon
from shapely.strtree import STRtree
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

use_entropy = False
do_cropping = True
# load images
# modesto - 20
# fresno - 412
# oxnard - 75
# stockton - 94
list_dirs = ['images_stockton', 'images_modesto','images_oxnard', 'images_fresno' ]
# source: https://figshare.com/articles/dataset/Distributed_Solar_Photovoltaic_Array_Location_and_Extent_Data_Set_for_Remote_Sensing_Object_Identification/3385780/4

def main():
    loading_start = time.time()
    load_data(list_dirs, use_entropy, do_cropping)
    loading_end = time.time()
    print ("Loading time: ", loading_end - loading_start)

    train_path = 'data_train/' if use_entropy else 'data_train_cropped256/';
    label_path = 'data_label/' if use_entropy else 'data_label_cropped256/';
    prediction_path = 'data_prediction/';
    if not do_cropping:
        train_path = 'data_train_no_cropping/';
        label_path = 'data_label_no_cropping/';
        prediction_path = 'data_prediction_no_cropping/';

    train_img_name = [os.path.basename(x) for x in glob(f'{train_path}*.tif')]
    label_img_name = [os.path.basename(x) for x in glob(f'{label_path}*.tif')]

    preprocessing_time = []
    detection_time = []

    for img_name, label_name in zip(train_img_name, label_img_name):

        preprocessing_start = time.time()
        ## PREPROCESSING
        original_img = cv2.imread(f'{train_path}'+img_name)
        copy_img = original_img.copy()
        # adjust contrast
        copy_img = apply_brightness_contrast(copy_img, 0, 75)
        # morphological operations
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.erode(copy_img,kernel,iterations = 1)
        # blur
        opening = cv2.medianBlur(opening, 3)
        #opening = cv2.dilate(opening,kernel,iterations = 1)

        # HSV
        hsv_img = cv2.cvtColor(opening, cv2.COLOR_BGR2HSV)
        BLUE_MIN = np.array([95, 90, 50],np.uint8)
        BLUE_MAX = np.array([125, 255, 255],np.uint8)
        dest = cv2.inRange(hsv_img, BLUE_MIN, BLUE_MAX)

        dest2 = cv2.cvtColor(dest,cv2.COLOR_GRAY2BGR)
        dest2 = cv2.erode(dest2,kernel,iterations = 1)

        preprocessing_end = time.time()

        ## DETECTION
        # MSER
        mser = cv2.MSER_create()
        # prediction
        regions = mser.detectRegions(dest2)

        detection_end = time.time()
        preprocessing_time.append(preprocessing_end - preprocessing_start)
        detection_time.append(detection_end - preprocessing_end)

        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
        img_mask = np.zeros(original_img.shape[:2], np.uint8)
        cv2.fillPoly(img_mask, hulls, 1)
        tiff.imsave(f'{prediction_path}'+img_name, img_mask)

    ## EVALUATION
    img_id = [os.path.basename(x) for x in glob(f'{train_path}*.tif')]

    l_actual = [tiff.imread(f'{label_path}'+x) for x in img_id]
    l_predicted = [tiff.imread(f'{prediction_path}'+x) for x in img_id]
    
    l_actual_col = [x.reshape(-1) for x in l_actual]
    l_predicted_col = [x.reshape(-1) for x in l_predicted]
    
    y_true = np.concatenate(l_actual_col, axis=0)
    y_pred = np.concatenate(l_predicted_col, axis=0)


    print ("Preprocessing time: ", np.sum(preprocessing_time))
    print ("Detection time: " , np.sum(detection_time))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    print ("Confusion matrix: " , cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Solar Panels','None'])
    disp.plot(cmap=plt.cm.Greens)
    plt.title("Confusion Matrix")
    plt.show()
    cm_norm = confusion_matrix(y_true, y_pred, labels=[1, 0], normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['Solar Panels','None'])
    disp.plot(cmap=plt.cm.Blues, values_format='.5f')
    plt.title("Normalized Confusion Matrix")
    plt.show()

    # Detection report
    print(classification_report(y_true, y_pred, target_names=['None', 'Solar panels']))

    # Intersection over Union
    total_jaccard = []
    for img_act, img_pred, img_name  in zip(l_actual, l_predicted, img_id):
        act_contour = cv2.findContours(img_act, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        act_contour = act_contour[0] if len(act_contour) == 2 else act_contour[1]
        if (len(act_contour[0]) == 1):
            continue
        if (len(act_contour[0]) == 2):
            continue
        # img = tiff.imread(f'{train_path}'+img_name)
        # img_copy = img.copy()
        # cv2.drawContours(img, act_contour, -1, (0,255,0), 3)
        
        pred_contour = cv2.findContours(img_pred, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        pred_contour = pred_contour[0] if len(pred_contour) == 2 else pred_contour[1]

        # cv2.drawContours(img_copy, pred_contour, -1, (255,0,0), 3)
        # h_img = cv2.hconcat([img, img_copy])
        # cv2.imshow('Detection',h_img)
        # cv2.waitKey(0)

        polys_act = [Polygon(np.squeeze(a_c)) for a_c in act_contour if (len(a_c) > 2)]
        s = STRtree(polys_act)

        for j, rect_b in enumerate(pred_contour):
            b = Polygon(np.squeeze(rect_b))
            
            for a in s.query(b):
                intersection_area = a.intersection(b).area
                if intersection_area:
                    total_jaccard.append(intersection_area / a.union(b).area)
    print("Jaccard score min: ", np.min(total_jaccard))
    print("Jaccard score max: ", np.max(total_jaccard))
    print("Jaccard score mean: ", np.mean(total_jaccard))
    print("Jaccard score median: ", np.median(total_jaccard))


if __name__ == "__main__":
    main()
