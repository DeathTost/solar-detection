import cv2
import numpy as np
import os, sys
import time
import argparse
import PIL
from PIL import Image
import shutil 
import matplotlib.pyplot as plt


# Wyszukiwanie obiektów na obrazie
def ProcessPhotos(directoryIN, directoryOUT, countStarter = 1):
    sum_all = 0
    sum_panels = 0
    # count_file = 0
    for file in os.listdir(directoryIN):
        # if file[-4:] == '.png':
        #     if count_file > 5000:
        #         return None
        #     count_file += 1
            image = Image.open(os.path.join(directoryIN, file))
            sciezka = os.path.join(directoryIN, file)
            path = file

            np.random.seed(10)

            CONFIDENCE = 0.4
            THRESHOLD = 0.2

            print('[INFO] Loading labels...')
            labels_path = r'C:\Users\Admin\Desktop\darknet-master\build\darknet\x64\data\obj.names'
            LABELS = open(labels_path).read().strip().split('\n')

            COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

            weights_path = r'C:\Users\Admin\Desktop\darknet-master\build\darknet\x64\yolov4-obj_last.weights'
            config_path = r'C:\Users\Admin\Desktop\darknet-master\build\darknet\x64\cfg\yolov4-obj.cfg'

            net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

            image = cv2.imread(sciezka)
            #cv2.imshow('img', image)
            (h, w) = image.shape[:2]

            ln = net.getLayerNames()
            ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            blob = cv2.dnn.blobFromImage(image, 1/255., (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layer_outputs = net.forward(ln)
            end = time.time()

            boxes = []
            confidences = []
            class_ids = []

            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > CONFIDENCE:
                        box = detection[0:4] * np.array([w, h, w, h])
                        (x_center, y_center, width, height) = box.astype('int')


                        x = int(x_center - (width / 2))
                        y = int(y_center - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

            if len(idxs) > 0:
                for i in idxs.flatten():

                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    color = [int(c) for c in COLORS[class_ids[i]]]

                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = f'{LABELS[class_ids[i]]}: {confidences[i]:.4f}'
                    cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Zapis pliku png z bounding boxes i confidance
            count = countStarter
            newFileName = os.path.split(file)[1] + '_{0}'.format(count) + '.png'
            if 0 in class_ids:
                cv2.imwrite(os.path.join(directoryOUT, newFileName), image)
            


p = r'C:\Users\Admin\Desktop\darknet-master\build\darknet\x64\data\obj\test'
w = r'C:\Users\Admin\Desktop\darknet-master\build\darknet\x64\data\obj\test\wyn'
ProcessPhotos(p, w)