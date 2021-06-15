## Dataset: Fresno (11ska), Stockton (10sfh), Oxnard (6..), and Modesto (10sfg)

import json
import cv2
import skimage
import matplotlib.pyplot as plt
import os.path
with open(r'D:\Doktorat_zajecia\CV\PROJECT\data_1\SolarArrayPolygons.json', 'r') as j:
    json_data = json.load(j)

image_name_list = []
for i in json_data['polygons']:
    if i['image_name'] in image_name_list:
        continue
    image_name_list.append(i['image_name'])

dx = -256
dy = -256
hh = 512
ww = 512

for n in image_name_list:
    x_all = []
    y_all = []
    polygon_list = []
    key = 'image_name'
    for i in json_data['polygons']:
        if i[key] == str(n):
            polygon_list.append(i['polygon_vertices_pixels'])

    for k in polygon_list:
        lista_x =[]
        lista_y =[]
        for l in range(len(k)):
            x = k[l][0]
            lista_x.append(x)
            y = k[l][1]
            lista_y.append(y)
            if len(lista_x) == len(k):
                x_all.append(lista_x)
                y_all.append(lista_y)
                break

    centrum_x = []
    centrum_y = []
    width_list = []
    height_list = []

    for x in x_all:
        center_x = min(x) + 0.5*(max(x) - min(x))
        centrum_x.append(center_x)

    for y in y_all:
        center_y = min(y) + 0.5*(max(y) - min(y))
        centrum_y.append(center_y)

    for w in x_all:
        width = max(w) - min(w)
        width_list.append(width)

    for h in y_all:
        height = max(h) - min(h)
        height_list.append(height)

    boxes = []
    for i in range(len(centrum_x)):
        box = [centrum_x[i], centrum_y[i], width_list[i], height_list[i]]
        boxes.append(box)

    count = 1
    cen_x = 0.5
    cen_y = 0.5
    name = str(n)
    path = f'D:/Doktorat_zajecia/CV/PROJECT/data/{name}.tif'
    if os.path.isfile(path):
        image = cv2.imread(path)
        #plt.imshow(image)
        counter1 = 1
        for box in boxes:
            #plt.plot(box[0], box[1], "or", markersize=2)
            cx = box[0]
            cy = box[1]
            left = cx + dx
            top = cy + dy
            left = int(cx + dx)
            top = int(cy + dy)
            if top <=  0 or left <= 0:
                continue
            startpoint = (int(box[0]-(0.5*box[2])),int(box[1]-(0.5*box[3])))
            endpoint = (int(box[0]+(0.5*box[2])),int(box[1]+(0.5*box[3])))
            #color = (255,0,0)
            #image100 = cv2.rectangle(image, startpoint, endpoint, color, thickness = 3)
            #image = cv2.circle(image, (int(cx), int(cy)), radius = 5, color = (255, 0, 0), thickness = 2)
            img_cropped = image[top:top+hh, left:left+ww]
            if img_cropped.shape != (512,512,3):
                continue
            #cv2.imwrite(f'D:\Doktorat_zajecia\CV\PROJECT\cropped do sprawdzenia\{name}_crop{count}.png', img_cropped)
            f = open(f'D:\Doktorat_zajecia\CV\PROJECT\Annotations\{name}_crop{counter1}.txt','w+')
            w_new = (box[2])/512
            h_new = (box[3])/512
            f.write(f'0 {cen_x} {cen_y} {w_new} {h_new}\n')
            f.close()
            print(f'Done for image {name}_crop{counter1}')
            count += 1
            counter1 +=1