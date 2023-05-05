import numpy as np
import pandas as pd
import cv2
# image processing imports
import skimage.io as io
from skimage.transform import resize

from skimage.feature import hog

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


def prepareData(images, labels):
    df = pd.DataFrame(images)
    df['Target'] = labels
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return np.asarray(x), np.asarray(y)


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
    Features = []
    labels = []

    for gender in ["men", "Women"]:
        datadir = r"Dataset\{}".format(gender)
        # loop over gender
        for hand in os.listdir(datadir):
            # loop over each class [0,1,2,3,4,5]
            for img in os.listdir(datadir + "/" + str(hand)):
                # ignoring anything except images
                if((img.split('.')[-1]).lower() not in ['jpg', 'png', 'jpeg']):
                    continue

                # loading our images
                # approx 2500 * 4000
                img_array = io.imread(
                    datadir + "/" + str(hand) + "/" + img, as_gray=True)

                # append extracted features to Featurees list
                Features.append(FeatureExtraction(img_array))

                # append class of image.
                labels.append(hand)

    return np.asarray(Features), np.asarray(labels)


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

    # downscaing from approx 2500x4000 to 500x500
    resized_image = resize(image, (500, 500))

    # Extract the hog features
    # block_norm uses L2 norm with hysterisis for reducing effect of illuminacity
    # transform_sqrt for applying gamma correction
    hog_features, hog_image = hog(resized_image, block_norm='L2-Hys',
                                  feature_vector=True, transform_sqrt=True, visualize=True)

    # image = np.array(resized).flatten() # flatten our image to be used as input vector to our model

    return hog_features, hog_image


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
    # downscaing from approx 2500x4000 to 500x500
    resized_image = resize(image, (500, 500))

    preprocessed_image = resized_image

    '''
        TODO: Preprocessing code should be implemented here.  
    '''

    return preprocessed_image


def SIFT(image):
    '''
        Inputs:
            image: the image to be processed
        Outputs:
            keypoints: the keypoints detected in the image
            descriptors: the descriptors of the keypoints
        Note: 
            to match keypoints of two images, we se a function called BFMatcher
            which takes two descriptors and match them together.
                ~ matches = cv2.BFMatcher(descriptor1, descriptor2)
        algorithm Explaination:
            this is a feature extraction technique used to detect objects even if they were 
            rotated, scaled or translated.
            the algorithms is built on 4 main steps:
            1. Constructing the Scale space
                This happens by two main steps:
                1.1- applying Gaussian filter to the image with different sigma values
                with the change of sigma, the blur of the image increases.
                and we also downScale image by factor = 2, this means that we generate octave with dimension
                equal to half the dimension of the previous octave.
                we repeat this process for certain number of levels, but they are typically 4.
                so we finally have 4 levels (octaves) and each level has 5 scales (sigma values). 
                1.2- now we apply (DoG) Difference of Gaussian on each octave, this step is very important in extracting 
                and enhancing the features of the image.
                it generates a set of images by subtracting each octave from the previous one.
                so this operation is applied for all octaves.
            2. Keypoint Localization:
                in this step we detect the local maxima and minima in the DoG images, and we also eliminate the low contrast
                keypoints by comparing them with a threshold.
                we also eliminate the keypoints that are on the edges by calculating the Hessian matrix of each keypoint and 
                checking if the ratio of the eigenvalues of this matrix is less than a certain threshold.
                    2.1 find local maximas and minimas by going through each pixel and comparing it with its 26 neighbors
                    26 as there are 8 direct neighbours in the same level, and 9 in the upper level and 9 in the lower level.
                    2.2 eliminate low contrast keypoints by comparing them with a threshold usually 0.03 in magnitude(there 
                    is a mathimatical proof for this step, it is interesting to have a look at :)). 
            3. Orientation Assignment:
                in this step we want to make the image orientation invariant, so we assign orientation to each keypoint
                we can do so by using 2 main steps:
                    3.1. Calculate magnitude and orientation
                        we calculate magnitude and orientation for each pixel in the image by applying sobel filter.
                        this happens by defining 3x3 window, and at each center pixel, we (x, y-1) from (x,y+1) to evaluate Gy
                        and then subtract (x-1, y) from (x+1, y) to get Gx, then we calculate magnitude and orientation.
                        mag = sqrt(Gx^2 + Gy^2), while direction is tan^-1(Gy/Gx). 
                        The magnitude represents the intensity of the pixel and the orientation gives the direction for the same.
                    3.2. create a histogram for magnitude and orientation
                        we create a histogram for each keypoint, and we divide the orientation range into 36 bins, so each bin
                        represents 10 degrees, and we add the magnitude of each pixel to the corresponding bin.
                        , the histogram will have a peak, so this peak represents the orientation of the keypoint, and we can also store
                        other peaks such as the peaks that are 80% of the maximum peak.
            4. Keypoint Descriptor:
                this is the final step of the algorithm, in this step we create a descriptor for each keypoint, so we can match
                the keypoints in different images.
                The descriptor is a unique fingerprint for each keypoint generated from the orientation, magnitude, and the neighbours
                of each pixel.
                they are the keys by which we detect the objects. 
        Reference:
            https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/


     '''

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def matching(des1, des2):
    '''
        Inputs:
            des1: the descriptors of the first image
            des2: the descriptors of the second image
        Outputs:
            matches: the matches between the two images
        Note:
            this function is used to match the keypoints of two images together.
    '''
    matches = cv2.BFMatcher(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def main():
    image = io.imread('hand.jpg', as_gray=True)
    hog_features, hog_image = FeatureExtraction(image)
    io.imshow(hog_features)


if __name__ == '__main__':
    main()
