import numpy as np
import cv2
from shapely.geometry import Polygon
import tifffile as tiff
from skimage.filters.rank import entropy
from skimage.morphology import rectangle 
import os
import json
from glob import glob
from functools import reduce

def read_geojson(data):
    """
    this function returns a dictionary with 
    key=image_name
    value=list of polygons coordinates, each polygon being a solar panel
    """
    d = dict()
    for i in range(len(data['features'])):
        im_name = data['features'][i]['properties']['image_name']
        if not im_name in d.keys():
            d[im_name] = []
        d[im_name].append(i)
    return(d)

def create_mask(d, data, img_path, img_id):
    """a
    this function creates a mask image with white pixels (1) in the solar panels
    and black pixels (0) everywhere else
    """
    img = tiff.imread(img_path)
    img_size = img.shape[:2]
    img_mask = np.zeros(img_size, np.uint8)
    
    try:
        ind_im = d[img_id]
    except KeyError:
        return(img_id, img, img_mask, [], []) # there are no solar panels in that image
    
    polys=[data['features'][i]['properties']['polygon_vertices_pixels'] for i in ind_im]
    lengths = [len(p) for p in polys]
    if 2 in lengths: del polys[lengths.index(2)]
        
    polygons = [Polygon(p) for p in polys]
    areas = [p.area for p in polygons]
    
    int_coords = lambda x: np.rint(np.array(x)).astype(np.int32)

    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [pi.coords for poly in polygons for pi in poly.interiors]

    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    solar_PV = img[img_mask==1,:]
    
    return(img_id, img, img_mask, areas, solar_PV)

def crop_image_entropy(img, mask, width, height):
    im3 = entropy(255*mask, rectangle(width, height))
    try:
        coord_x = max(np.where(im3==np.max(im3))[0][0]-width//2,0)
        coord_y = max(np.where(im3==np.max(im3))[1][0]-height//2,0)
    except:
        coord_x = 0
        coord_y = 0
    if coord_x+width>mask.shape[0]: coord_x = mask.shape[0]-width
    if coord_y+height>mask.shape[1]: coord_y = mask.shape[1]-height
    cropped_label = mask[coord_x:coord_x+width, coord_y:coord_y+height]
    cropped_img = img[coord_x:coord_x+width, coord_y:coord_y+height, :]
    return cropped_img, cropped_label

def crop_image_random(img, mask, x, y):
    cropped_img = img[y*256:(y+1)*256, x*256:(x+1)*256, :]
    cropped_label = mask[y*256:(y+1)*256, x*256:(x+1)*256]
    return cropped_img, cropped_label

def load_data(list_dirs, use_entropy, do_cropping):
    ## DATA LOADING
    # load geojson
    with open('images\\SolarArrayPolygons.geojson') as f:
        data = json.load(f)
        d = read_geojson(data)
    
    images_paths = [glob('images\\'+path+'/*.tif') for path in list_dirs]
    images_paths = reduce(lambda x,y: x+y, images_paths)
    
    # create masks
    images_id = [os.path.splitext(os.path.basename(im))[0] for im in images_paths]

    for img_p, img_id in zip(images_paths, images_id):
        _, img, mask, _, _ = create_mask(d, data, img_p, img_id)

        if not do_cropping:
            tiff.imsave('data_train_no_cropping/'+img_id+'.tif', img)
            tiff.imsave('data_label_no_cropping/'+img_id+'.tif', mask)
            continue;
        
        if use_entropy:
            cropped_img, cropped_label = crop_image_entropy(img, mask, 256, 256)
            tiff.imsave('data_train/'+img_id+'.tif', cropped_img)
            tiff.imsave('data_label/'+img_id+'.tif', cropped_label)
        else: 
            height, width = img.shape[:2]
            x_crops, y_crops = width//256, height//256
            for x in range(x_crops):
                for y in range(y_crops):
                    cropped_img, cropped_label = crop_image_random(img, mask, x, y)
                    if 1 in cropped_label and cropped_label.shape==(256, 256):
                        tiff.imsave('data_train_cropped256/'+img_id+'_'+str(x)+'_'+str(y)+'.tif', cropped_img)
                        tiff.imsave('data_label_cropped256/'+img_id+'_'+str(x)+'_'+str(y)+'.tif', cropped_label)