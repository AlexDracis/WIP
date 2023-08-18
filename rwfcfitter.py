############################################
#######   HarvestAi RFC measurement  #######
######## Last edited on 17.Aug.2023 ########
########    by Alexandre Sicard     ########
############################################

#Run the script to analyse a picture with an already trained model
#Generate csv files based on the number of classes defined
#TO DO: finish batching process
#TO DO: Incorporate more properties to the measurement
import os
import cv2 as cv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from functools import partial
from skimage import data, filters, measure, morphology
from skimage import data, segmentation, feature, future
import joblib


#RFC classification script
def classify(image: str):
    file = cv.imread("rfcfitter/data/" + image + ".jpg")
#Create a repository to save data
##    if not os.path.exists("rfcfitter/measurement".format(image)):
##     os.makedirs("rfcfitter/measurement".format(image))
#Define parameters for the classifier to be run. This must map the parameters used for the model training.->
    sigma_min = 1
    sigma_max = 3
    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=True, texture=True,
                            sigma_min=sigma_min, sigma_max=sigma_max, #num_sigma=10,
                            channel_axis=-1)
#Load the model
    Clf = joblib.load('rfcfitter/rf-model/rf-model.pkl')
#Run the classification
    features = features_func(file)
    result = future.predict_segmenter(features, Clf)
#Possibility to save picture with pixel intensity corresponding to the classification.
    cv.imwrite('rfcfitter/data/{}classes.png'.format(image), result)
#Prepare arrays for object detection and measurement    
    mask=np.empty(len(result), dtype = object)
    props=np.empty(len(result), dtype = object)
    table=np.empty(len(result), dtype = object)
    table2=np.empty(len(result), dtype = object)
    labels=np.empty(len(result), dtype = object)
#Run object detection and measurement for the defined classes. Must map the model training ->    
    for i in range (1,5):
        
#mask generation from each class of the classifier
        mask[i]=(result==i)
#Size exclusion filter. REQUIRE REWORK.
        mask[i] = morphology.remove_small_objects(mask[i],7000)
        mask[i] = morphology.remove_small_holes(mask[i], 7000)
    
#Detection of the ROI based on the mask     
        labels[i] = measure.label(mask[i])
#Measurement of properties of ROI (labels) on the original picture.
        props[i] = measure.regionprops(labels[i], file)
#Definition of properties of interest:
#  'num_pixels', 'area', 'area_bbox', 'area_convex', 'area_filled', 'axis_major_length', 'axis_minor_length', 'bbox', 'centroid', 'centroid_local', 'centroid_weighted', 'centroid_weighted_local', 'coords', 'eccentricity', 'equivalent_diameter_area', 'euler_number','extent',  'feret_diameter_max', 'image', 'image_convex', 'image_filled', 'image_intensity','inertia_tensor', 'inertia_tensor_eigvals', 'intensity_max', 'intensity_mean', 'intensity_min', 'label', 'moments', 'moments_central', 'moments_hu', 'moments_normalized', 'moments_weighted', 'moments_weighted_central', 'moments_weighted_hu', 'moments_weighted_normalized', 'orientation', 'perimeter', 'perimeter_crofton', 'slice', 'solidity'  
        properties = ['area' ,'bbox', 'centroid','coords', 'eccentricity', 'euler_number','extent', 'intensity_max', 'intensity_mean', 'intensity_min', 'label', 'perimeter', 'solidity']
#Generation of a file to record labels properties
        table[i]=measure.regionprops_table(labels[i],file,properties)
        
        table2[i]=pd.DataFrame(table[i])
        table2[i].to_csv("rfcfitter/measurement/classes{}.csv".format([i]))
    
    return

def list_files():
    images = os.opendir("rfcfitter/data/*.jpg")
    return images
