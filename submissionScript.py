# ML models imports
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.transform import probabilistic_hough_line
from skimage.feature import canny
from xgboost import XGBClassifier
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import time

# destribution models imports
from scipy.stats import randint, uniform

# Data manipulation imports
import numpy as np
import pandas as pd
import tqdm as tqdm

# visualisation models imports
import matplotlib.pyplot as plt

# image processing imports
import skimage.io as io
import cv2
from skimage.transform import resize

# dealing with files
import os

# visual dataset (to test randomized gridsearch not needed for now)
from sklearn.datasets import make_hastie_10_2  # to test our models

# from utils import prepareData, LoadData, FeatureExtraction, preprocess
import csv
import joblib




def segment(image):
    blured_image = cv2.GaussianBlur(image, (7, 7), 0)
    ycbcr_image = cv2.cvtColor(blured_image, cv2.COLOR_BGR2YCrCb)
    # Extract the Cr channel
    cr_channel = ycbcr_image[:, :, 1]

    # Apply thresholding to obtain a binary image
    _, binary_img = cv2.threshold(
        cr_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Define the structuring element for the closing operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))

    # Perform the closing operation
    closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    # Find the contours in the binary image
    contours, hierarchy = cv2.findContours(
        closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filling the contours on a copy of the original image
    # img_contours = cv2.cvtColor(cr_channel, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(img_contours, contours, -1, (0, 0, 0), -1)

    segmented_image = closed_img.copy()
    cv2.drawContours(segmented_image, contours, -1, 255, -1)

    return segmented_image

def Blobs(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian of Gaussian (LoG) filter
    log = cv2.GaussianBlur(gray, (5, 5), 0)
    log = cv2.Laplacian(log, cv2.CV_8U, ksize=5)

    # Threshold the filtered image to obtain a binary image
    _, binary = cv2.threshold(log, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Apply morphological operations to remove noise and fill gaps in the blobs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Detect and extract the blobs using the SimpleBlobDetector in OpenCV
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 255
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(binary)

    extracted_blobs = []

    for kp in keypoints:
        f = extract_from_keypoint(kp)
        extracted_blobs.append(f)

    return np.asarray(extracted_blobs).flatten()

def numStraightLines(image):
    edges = canny(image, 2, 1, 25)
    lines = probabilistic_hough_line(
        edges, threshold=10, line_length=5, line_gap=3)
    return len(lines)

def HUFunction(img):
    '''
        this is a utility function used to evaluate the HU features which 
        are 7, which are invarient with size, rotation and transition. 
        inputs:
            img: the original Img
        outputs:
            the huMoments. 
    '''
    # Calculate Moments
    moments = cv2.moments(img)
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)

    return huMoments


def FeatureExtraction(image):
    
    Hu_M = HUFunction(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    preprocessed_image = segment(image)

    resized_image = resize(preprocessed_image, (64, 128))

    hog_features = hog(resized_image, block_norm='L2-Hys', feature_vector=True,
                    transform_sqrt=True, pixels_per_cell=(12, 12), cells_per_block=(2, 2))

    features = np.concatenate(((Hu_M.flatten()).reshape(-1,1), hog_features.reshape(-1,1)), axis=0)

    return np.squeeze(features)


def LoadData():
    Features = []
    labels = []

    i = 0
    for gender in ["men", "Women"]:
        datadir = r"dataset_no3\{}".format(gender)
        # loop over gender
        for hand in os.listdir(datadir):
            # loop over each class [0,1,2,3,4,5]
            for img in os.listdir(datadir + "/" + str(hand)):
                # ignoring anything except images
                if ((img.split('.')[-1]).lower() not in ['jpg', 'png', 'jpeg']):
                    continue

                # loading our images
                # approx 2500 * 4000
                img_array = io.imread(datadir + "/" + str(hand) + "/" + img)

                # append extracted features to Featurees list
                Feature = FeatureExtraction(img_array)
                Features.append(Feature)

                # append class of image.
                labels.append(int(hand))

                print(f'image Number: {i}')
                i += 1

    return np.asarray(Features), np.asarray(labels)


def testOurModel(path):
    '''
       this is a utility function used to load the test images
    '''

    y_pred = []
    datadir = path
    svm = joblib.load('model_2_All_Images_HOG_HU.pkl')
    # loop over gender

    for img in sorted(os.listdir(datadir), key=lambda x: int(x.split('.')[0])):
        # ignoring anything except images
        if((img.split('.')[-1]).lower() not in ['jpg', 'png', 'jpeg']):
            continue
    
        tic = time.time()
        y_pred = logic(svm,io.imread(datadir + "/" + img))
        toc = time.time()

        with open("time.txt", "a") as file:
            file.write(f'{(toc-tic)*1000} ms\n')
            

        with open("results.txt", "a") as file:
            file.write(f'{y_pred[0]}\n')

    return



def logic(svm,image):
    features = FeatureExtraction(image)
    y_pred = svm.predict(features.reshape(1,-1))

    return y_pred



def main():
    # Main logic of the script
    path = input("Please enter image path : ")
    testOurModel(path)



# Call the main() function if this script is being run directly
if __name__ == "__main__":
    main()
