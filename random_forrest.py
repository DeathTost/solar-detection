### Source code credit https://github.com/alexhalcomb/DSI-Capstone-Project


import pandas as pd
import numpy as np
import gdal
from scipy.stats import describe

from skimage.segmentation import quickshift, mark_boundaries 
from skimage.draw import polygon, polygon_perimeter
from skimage import measure
from skimage import io

from matplotlib.patches import Circle, Polygon
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from ipywidgets import interact, fixed
from os import listdir
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, precision_recall_curve, classification_report
from sklearn.metrics import roc_curve, auc

# Function to clean polygon vertices data 
def get_cords(x):
  lon=[]
  lat=[]
  for i in range(int(x[1]-1)):
    lon.append(int(x[2+2*i]))
    lat.append(int(x[3+2*i]))
  return list(zip(lat,lon))

def extract_vertices(df):
  df['polygon_vertices']=df.apply(lambda x :get_cords(x),axis=1)
  return df[['polygon_id','polygon_vertices']]

# Plot centers of all solar arrays for a given satellite image

def plot_centroids(img_id):
    
    # Use img_id to create path to correct directory 
    #city = arrays_clean[arrays_clean['image_name'] == img_id]['city'].values[0]
   # city = 'Oxnard ' + city
    
    # Read in image
    img = io.imread('/content/gdrive/MyDrive/CV/%s.tif' % (img_id))
    
    # Extract centroid coordinates from dataframe    
    arrays_sub = arrays_clean[arrays_clean['image_name'] == img_id]
    coordinates = zip(list(arrays_sub['centroid_longitude_pixels']),
                      list(arrays_sub['centroid_latitude_pixels']))

    # Display image
    fig,ax = plt.subplots(figsize=(12, 12))

    ax.imshow(img[:,:,1], cmap=cm.gray)
    
    # Plot a circle for each solar panel array
    for x, y in coordinates:
        centroid = Circle((x, y), 20, color='blue')
        ax.add_patch(centroid)
 

# Plot polygon outline for a given solar panel array

def plot_polygon(poly_id):
     
    # Use poly_id to identify image name and create path to correct directory 
    img_id = arrays_clean[arrays_clean['polygon_id'] == poly_id]['image_name'].values[0]
    city = arrays_clean[arrays_clean['image_name'] == img_id]['city'].values[0]
    city = 'Oxnard ' + city
        
    # Extract polygon vertices from dataframe, calculate min/max
    vertices = arrays_clean[arrays_clean['polygon_id'] == poly_id][['polygon_vertices']].iloc[0,0]
    
    x_values = [x[0] for x in vertices]
    y_values = [y[1] for y in vertices]
    x_min, x_max = min(x_values), max(x_values) 
    y_min, y_max = min(y_values), max(y_values)

    # Read in image, crop   
    img = io.imread('/content/gdrive/MyDrive/CV/%s.tif' % (img_id))
    band = 20
    img_cropped = img[x_min - band : x_max + band, 
                      y_min - band : y_max + band]

    # Display cropped image
    fig,ax = plt.subplots()

    ax.imshow(img_cropped)

    # Plot a polygon for each solar panel array. Vertices must be flipped to conform to 'Polygon' convention
    resized_coords = [(y - (y_min - band) - 1, x - (x_min - band) - 1) for x, y in vertices]
    poly = Polygon(resized_coords, fill=False, color='blue')
    ax.add_patch(poly)


# Extract spatial features

def segment_features_shape(segments, img_id='na', quickshift=False):

    # Skimage segmentation algorithms assign a "0" label, however, measure.regionprops only reads labels > 0 
    # If quickshift == True, add 1 to all segment labels
    if quickshift:
        segments = segments + 1
        
    # Convert labels to int and generate unique labels list
    segments = segments.astype(int)
    segment_labels = np.unique(segments[segments > 0])

    # For each region/segment, create regionprops object and extract features
    region_props_all = measure.regionprops(segments)    
    region_features = {}
        
    for i, region in enumerate(region_props_all):    

        shape_features = {}

        shape_features['image_id'] = img_id
        shape_features['segment_id'] = segment_labels[i]
        shape_features['perimeter'] = region.perimeter
        shape_features['area'] = region.area
        shape_features['circleness'] = (4 * np.pi * region.area) / (max(region.perimeter, 1) ** 2)
        shape_features['centroid'] = region.centroid
        shape_features['coords'] = region.coords

        region_features[segment_labels[i]] = shape_features
    
    index = ['image_id', 'segment_id', 'perimeter', 'area', 'circleness', 'centroid', 'coords']
    
    shape_df = pd.DataFrame(region_features, index=index).T
    
    return shape_df
  
# Extract color features

def segment_features_color(img, segments, img_id='na', quickshift=False):
    
    # Skimage segmentation algorithms assign a "0" label, however, measure.regionprops only reads labels > 0 
    # If quickshift == True, add 1 to all segment labels
    if quickshift:
        segments = segments + 1
        
    # Convert labels to int and generate unique labels list
    segments = segments.astype(int)
    segment_labels = np.unique(segments[segments > 0])

    # For each segment and channel, calculate summary stats    
    channels = ['r','g','b']    
    region_features = {}
    
    for label in segment_labels:
        region = img[segments == label]
        
        color_features = {}        
               
        for i, channel in enumerate(channels):
            values = describe(region[:,i])
            
            color_features['image_id'] = img_id
            color_features['segment_id'] = label
            color_features[channel + '_min'] = values.minmax[0]
            color_features[channel + '_max'] = values.minmax[1]
            color_features[channel + '_mean'] = values.mean
            color_features[channel + '_variance'] = values.variance
            color_features[channel + '_skewness'] = values.skewness
            color_features[channel + '_kurtosis'] = values.kurtosis
            
        region_features[label] = color_features
        
    index = ['image_id','segment_id','r_min','r_max','r_mean','r_variance','r_skewness',
         'r_kurtosis','g_min','g_max','g_mean','g_variance','g_skewness','g_kurtosis',
         'b_min','b_max','b_mean','b_variance','b_skewness','b_kurtosis']
        
    color_df = pd.DataFrame(region_features, index=index).T
        
    return color_df

# Create a mask for all solar panels in the image, with each labeled by its polygon_id

def panel_mask(img_id, use_labels=False):
  
    # Identify city in order to navigate to correct directory 
    city = arrays_clean[arrays_clean['image_name'] == img_id]['city'].values[0]
    #city = 'Duke ' + city
    
    
    # Read in image
   # img = io.imread('/Users/alexanderhalcomb/Desktop/SolarPV/Datasets/%s/%s.tif' % (city, img_id))
    img = io.imread('/content/gdrive/MyDrive/CV/%s.tif' % (img_id))

    # Select polygon vertices for all polygons in image
    vertices = arrays_clean[arrays_clean['image_name'] == img_id]['polygon_vertices'].values
    labels = arrays_clean[arrays_clean['image_name'] == img_id]['polygon_id'].values
    
    # Create scaffolding for mask
    mask = np.zeros((img.shape[0],img.shape[1]))
    mask[mask == 0] = np.nan
        
    # Option to use custom labels for each panel array
    if use_labels == False:
        poly_data = enumerate(vertices, 1)
    else:
        poly_data = zip(labels, vertices)
        
    # Iterate through vertices and assign labels to pixels
    for i, poly in poly_data:
        x = np.array([vert[0] for vert in poly])
        y = np.array([vert[1] for vert in poly])
        
        mask[polygon(x, y, shape=(img.shape[0],img.shape[1]))] = i 
        mask[polygon_perimeter(x, y,shape=(img.shape[0],img.shape[1]))] = i

    return mask
# Create a series of random windows that do not overlap with any solar panels

def no_panel_mask(panel_mask, window_size=100, windows=20):
    
    # Create scaffolding for mask 
    panels = panel_mask.copy() 
    
    mask = np.zeros(panels.shape)
    mask[mask == 0] = np.nan
        
    rows = panels.shape[0]
    cols = panels.shape[1]
        
    # Create i number of random windows with no panels
    for i in range(windows): 
    
        search = True
        
        # Iterate until random window found
        while search:

            random_x = np.random.randint(0, rows - window_size + 1)
            random_y = np.random.randint(0, cols - window_size + 1)
            random_window = panels[random_x: random_x + window_size, random_y: random_y + window_size]

            # End search when random window is all nan, i.e. does not overlap with any panels
            if np.isnan(random_window).sum() == random_window.size:
                
                # Update mask to include new random window
                mask[random_x: random_x + window_size, random_y: random_y + window_size] = i + 1
                
                # Update panel_mask copy to avoid duplicate random window on next iteration
                panels[random_x: random_x + window_size, random_y: random_y + window_size] = 0
                
                # Go to next window 
                search = False
                
    return mask

# Return masks

def return_masks(img_id):
    
    # Create panel mask 
    panels = panel_mask(img_id, use_labels=True)
    
    # Create no panel mask 
    no_panels = no_panel_mask(panels, window_size=60, windows=25)
    
    return panels, no_panels

# Return features for all solar panels

def extract_pos_features(img_id, quickshift=False):
    
    # Identify city in order to navigate to correct directory 
    city = arrays_clean[arrays_clean['image_name'] == img_id]['city'].values[0]
    city = 'Duke ' + city
    print(img_id)
    # Read in image
    #img = io.imread('/Users/alexanderhalcomb/Desktop/SolarPV/Datasets/%s/%s.tif' % (city, img_id))
    img = io.imread('/content/gdrive/MyDrive/CV/%s.tif' % (img_id))
    
    # Create panel mask 
    segments_panels = panel_mask(img_id, use_labels=True)
    
    # Extract shape features
    shape_df = segment_features_shape(segments_panels, img_id, quickshift=quickshift)
        
    # Extract color features
    color_df = segment_features_color(img, segments_panels, img_id, quickshift=quickshift)

    # Combine features into single dataframe
    features_df = shape_df.merge(color_df, how='inner', on=['image_id','segment_id'])
    
    return features_df

# Return features for regions without solar panels

def extract_neg_features(img_id, window_size=100, windows=20):
    
    # Identify city in order to navigate to correct directory 
    city = arrays_clean[arrays_clean['image_name'] == img_id]['city'].values[0]
    city = 'Duke ' + city
    
    # Read in image
    #img = io.imread('/Users/alexanderhalcomb/Desktop/SolarPV/Datasets/%s/%s.tif' % (city, img_id))
    img = io.imread('/content/gdrive/MyDrive/CV/%s.tif' % (img_id))
              
    # Create panel mask 
    segments_panels = panel_mask(img_id, use_labels=True)

    # Create no panel mask 
    windows_no_panels = no_panel_mask(segments_panels, window_size=window_size, windows=windows)
    windows_labels = np.unique(windows_no_panels[windows_no_panels > 0])

    frames = []
    
    for label in windows_labels:

        # Extract window from image
        window_img = img[windows_no_panels == label].reshape((window_size, window_size, 3))
        
        # Divide window into segments using Quickshift segmentation algorithm 
        segments_quick = quickshift(window_img, kernel_size=3, max_dist=8, ratio=0.65)
        segments_labels = np.unique(segments_quick)

        # Extract shape features
        shape_df = segment_features_shape(segments_quick, img_id, quickshift=True) 

        # Extract color features
        color_df = segment_features_color(window_img, segments_quick, img_id, quickshift=True)

        # Combine features into single dataframe
        features_df = shape_df.merge(color_df, how='inner', on=['image_id','segment_id'])
        
        # Append to list to facilitate aggregation
        frames.append(features_df)
        
    return pd.concat(frames)
 
# Input image(s) and feed through entire pipeline, return df of results

def process_images(images):
    
    frames = []
    
    for img_id in images:
        
        pos_features = extract_pos_features(img_id)
        pos_features['panel_class'] = 1

        neg_features = extract_neg_features(img_id, window_size=60, windows=25)
        neg_features['panel_class'] = 0

        combined_features = pd.concat([pos_features, neg_features])
        combined_features.dropna(inplace=True)
        
        frames.append(combined_features)
    
    return pd.concat(frames).reset_index(drop=True)


# Function to plot precision recall curve

def plot_precision_recall(y_true, y_prob, title='Precision Recall Curve'):
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7,5))

    plt.plot(recall, precision, 'b')
    plt.plot([1,0], [0,1], '--r')
    
    plt.title(title, fontsize=16)   
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)

    plt.show()
   
# Function to plot ROC curve

def plot_roc(y_true, y_prob, title='Receiver Operating Characteristic'):
    
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)

    fig, ax = plt.subplots(figsize=(7,5))

    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.3f'% roc_auc)
    plt.plot([0,1],[0,1],'r--')
    
    plt.title(title, fontsize=16)
    plt.xlabel('False Positive Rate (Fall-out)', fontsize=14)
    plt.ylabel('True Positive Rate (Recall)', fontsize=14)
    plt.legend(loc='lower right')

    plt.show()
# Function to plot confusion matrix, normalized=True returns cm in the form of percentages

def plot_confusion_matrix(y_true, y_pred, cmap=plt.cm.Greens, normalize=False, title='Confusion Matrix'):
    
    matrix = confusion_matrix(y_true, y_pred, labels=[1,0])

    fig, axes = plt.subplots(figsize=(5,5))
    fig.tight_layout()
    
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        plt.title(title, fontsize=16)
    else:
        plt.title(title, fontsize=16)
        
    classes = ['Solar Panels','None']
    ticks = np.arange(len(classes))

    plt.imshow(matrix, interpolation='nearest', cmap=cmap)

    thresh = matrix.max() / 2.
    
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            plt.text(col, row, np.around(matrix[row, col], decimals=3), horizontalalignment='center', 
                     color='white' if matrix[row, col] > thresh else 'black', fontsize=14)
    
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes, rotation=45)
    
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    
    plt.colorbar()
    plt.show()

raw_data_1 = ('/content/gdrive/MyDrive/CV/polygonDataExceptVertices.csv')
raw_data_2 = ('/content/gdrive/MyDrive/CV/polygonVertices_PixelCoordinates.csv')

arrays = pd.read_csv(raw_data_1,index_col=0)
vertices_pix = pd.read_csv(raw_data_2)
vertices_pix_clean = extract_vertices(vertices_pix)
arrays_clean = arrays.merge(vertices_pix_clean, how='inner', on='polygon_id')
arrays_clean = arrays_clean[['polygon_id', 'centroid_latitude_pixels', 'centroid_longitude_pixels', 'city', 
                            'image_name', 'polygon_vertices']]
images=arrays_clean['image_name'].unique()
images=np.delete(images,np.where(images == '11ska385725'))
combined_features=process_images(images)

# Create X and y data

excluded = ['panel_class', 'image_id', 'segment_id', 'centroid', 'coords', 'r_skewness', 
            'g_skewness', 'b_skewness','r_kurtosis', 'g_kurtosis', 'b_kurtosis']

X_cols = [col for col in combined_features.columns if col not in excluded]

ss = StandardScaler()

X = combined_features[X_cols]
Xn = ss.fit_transform(X)

y = combined_features['panel_class']
y_base = max(np.mean(y), 1 - np.mean(y))

# Split data into a training and testing set

X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.3, stratify=y)

print (X_train.shape, X_test.shape)
print (y_train.shape, y_test.shape)

# GridSearch a RF classifier
import time
rf = RandomForestClassifier()

rf_params = {
    'max_features':[None,'log2','sqrt', 2,3,4,5],
    'max_depth':[1,2,3,None],
    'min_samples_leaf': np.arange(2,101,13),
    'n_estimators':[50]
}



rf_gs = GridSearchCV(rf, rf_params, cv=5, verbose=1, n_jobs=-1)

elapsed_time = time.time() - start_time



rf_gs.fit(X_train, y_train)



print ('Done fitting.')

# Check results of Random Forest GridSearch

rf_best = rf_gs.best_estimator_
print ('Random Forest GridsearchCV Results:')
print ('Best Estimator', rf_best)
print ('Best Parameters',  rf_gs.best_params_)
print ('Best Score', '%0.4f' % rf_gs.best_score_, '\n')

# Check results against hold out test set 

print ('Cross Validation on Test Set:')
print ('Score: ', '%0.4f' % rf_gs.score(X_test, y_test))
print ('Baseline Accuracy: ', '%0.4f' % y_base)
print ('Percent Better: ', '%0.4f' % ((rf_gs.score(X_test, y_test) - y_base) / (1 - y_base)))
# Extract RF feature importance

feature_importance = pd.DataFrame({'feature': X.columns,
                                   'importance': rf_best.feature_importances_})

feature_importance.sort_values('importance', ascending=False, inplace=True)

print(feature_importance)

# Plot RF feature importance

fig, ax = plt.subplots(figsize=(8, 6))

ax.barh(np.arange(len(feature_importance['importance'].values)), feature_importance['importance'], 0.8, align='center')

ax.set_yticks(np.arange(len(feature_importance['importance'].values)))
ax.set_yticklabels(feature_importance['feature'], fontsize=10)
ax.set_ylabel('Feature', fontsize=14)
ax.set_xlabel('Importance', fontsize=14)
ax.set_ylim([-0.5, 15.5])
ax.set_xlim([feature_importance['importance'].min()-0.025, feature_importance['importance'].max()+0.025])
ax.set_title('Random Forest Feature Importance', fontsize=16)
plt.show()

y_pred_rf = rf_gs.predict(X_test)
y_prob_rf = rf_gs.predict_proba(X_test)[:,1]
plot_confusion_matrix(y_test, y_pred_rf, title='Confusion Matrix')
plot_confusion_matrix(y_test, y_pred_rf, normalize=True, cmap=plt.cm.Blues, title='Normalized Confusion Matrix')
plot_roc(y_test, y_prob_rf, title= 'ROC Curve')
plot_precision_recall(y_test, y_prob_rf, title='Precision Recall Curve')
from sklearn.metrics import classification_report
print(classification_report(y_test2,y_pred_rf))
