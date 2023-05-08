import numpy as np
import pandas as pd

# image processing imports
import skimage.io as io
from skimage.transform import resize

from skimage.feature import hog

import cv2

# dealing with files
import os

'''
Input:
    images: a list or array-like object containing the input data, where each row represents an image
    labels: a list or array-like object containing the corresponding output labels for each image

Output:
    x: a pandas DataFrame containing the input data (images)
    y: a pandas Series containing the output (predicted) data (labels)

Functionality:
    This function takes two inputs, images and labels, and returns a tuple of input and output data,
    respectively. First, it converts the input data (images) into a pandas DataFrame,
    and then it adds a new column 'Target' to the DataFrame with corresponding output labels.
    The input data is then extracted from the DataFrame by selecting all the columns except the last column,
    which represents the output data. The output data is also extracted by selecting only the last column of the DataFrame.
    Finally, the input and output data are returned as x and y, respectively. this function is used for dealing with images 
    easily and smoothly.
Example:
    x, y = prepareData(images,labels) 

'''
def prepareData(images,labels):
    df=pd.DataFrame(images) 
    df['Target'] = labels
    x = df.iloc[:,:-1]  
    y = df.iloc[:,-1] 
    return np.asarray(x),np.asarray(y)



'''
Input:
    Gender: a list containing  

Output:
    Features: a list containing extracted features from the images
    labels: a list containing extracted labels

Functionality:
    This function is used to process data from their directory and extract features from the images.

Example:
    Features, labels = LoadData() 

Note: 
    [*] Structure of Dataset foulder should be as follows in the project directory:

                       Dataset
                    men       Women
              0 1 2 3 4 5    0 1 2 3 4 5
           
        -> each number is a directory contains the images of that class.

    [*] this structure existed initially in the given dataset i didn't came up with :) 



'''
def LoadData():
    Features=[]
    labels=[]

    for gender in ["men","Women"]:
        datadir = r"Dataset\{}".format(gender)
        # loop over gender
        for hand in os.listdir(datadir): 
            # loop over each class [0,1,2,3,4,5]
            for img in os.listdir(datadir+ "/" +str(hand)):
                # ignoring anything except images
                if((img.split('.')[-1]).lower() not in ['jpg','png','jpeg']):
                    continue

                # loading our images
                img_array=io.imread(datadir + "/" + str(hand) + "/" + img ,as_gray=True)  # approx 2500 * 4000

                # append extracted features to Featurees list           
                Features.append(FeatureExtraction(img_array)) 

                # append class of image.
                labels.append(hand)     

    return np.asarray(Features),np.asarray(labels)


'''
input:
    images: an image of hand.

output:
    Extracted features ready to be used in training.

Functionality: 
    This function is used for Extracting features from images.
Example:
    features = extractFeatures(image)
'''
def FeatureExtraction(image):
    # this written code is an initial code for extracting features

    '''

        TODO: Feature Extraction code should be implemented here.  

    '''
    
    resized_image = resize(image,(500,500))   # downscaing from approx 2500x4000 to 500x500
    
    # Extract the hog features
    # block_norm uses L2 norm with hysterisis for reducing effect of illuminacity
    # transform_sqrt for applying gamma correction
    segmented = segment(resized_image)

    hog_features = hog(resized_image, block_norm='L2-Hys', feature_vector=True, transform_sqrt=True)

    # image = np.array(resized).flatten() # flatten our image to be used as input vector to our model

    return hog_features

    
'''
input:
    image: The image to be preprocessed.

output:
    The preprocessed image.
    
note: 
    I think it may be more optimized to make the function take an array of images and preprocess all of them at once
    This will reduce the overhead of calling the function for each image.
'''
def preprocess(image):
    resized_image = resize(image,(500,500))   # downscaing from approx 2500x4000 to 500x500
    
    preprocessed_image = resized_image
    
    '''
        TODO: Preprocessing code should be implemented here.  
    '''
    
    return preprocessed_image

def segment(image):
    blured_image = cv2.GaussianBlur(image, (7, 7), 0)
    ycbcr_image = cv2.cvtColor(blured_image, cv2.COLOR_BGR2YCrCb)
    # Extract the Cr channel
    cr_channel = ycbcr_image[:,:,1]

    # Apply thresholding to obtain a binary image
    _, binary_img = cv2.threshold(cr_channel,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Define the structuring element for the closing operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

    # Perform the closing operation
    closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    # Find the contours in the binary image
    contours, hierarchy = cv2.findContours(closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filling the contours on a copy of the original image
    img_contours = cv2.cvtColor(cr_channel, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_contours, contours, -1, (0, 0, 0), -1)

    return img_contours


def main():
    image = io.imread('hand.jpg' ,as_gray=True)
    hog_features, hog_image = FeatureExtraction(image)
    io.imshow(hog_features)
    
    
if __name__ == '__main__':
    main()