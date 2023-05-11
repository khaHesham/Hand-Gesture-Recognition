import numpy as np
import pandas as pd
# sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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
    segmented = segment(resized_image)

    hog_features = hog(resized_image, block_norm='L2-Hys',
                       feature_vector=True, transform_sqrt=True)

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
    # downscaing from approx 2500x4000 to 500x500
    resized_image = resize(image, (500, 500))

    preprocessed_image = resized_image

    '''
        TODO: Preprocessing code should be implemented here.  
    '''

    return preprocessed_image


def segment(image):
    blured_image = cv2.GaussianBlur(image, (7, 7), 0)
    ycbcr_image = cv2.cvtColor(blured_image, cv2.COLOR_BGR2YCrCb)
    # Extract the Cr channel
    cr_channel = ycbcr_image[:, :, 1]

    # Apply thresholding to obtain a binary image
    _, binary_img = cv2.threshold(
        cr_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Define the structuring element for the closing operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

    # Perform the closing operation
    closed_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    # Find the contours in the binary image
    contours, hierarchy = cv2.findContours(
        closed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filling the contours on a copy of the original image
    img_contours = cv2.cvtColor(cr_channel, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_contours, contours, -1, (0, 0, 0), -1)

    return img_contours

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ZIZO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#


def featureScalingManually(features):
    '''
        Inputs:
            features: the features to be scaled
        Outputs:
            scaledFeatures: the scaled features
        Note: 
            this function is used to scale the features to be in the range [0,1]
            this is done by subtracting the minimum value from each feature and then dividing by the range
            of the feature.
    '''
    scaledFeatures = np.zeros(features.shape)
    for i in range(features.shape[1]):
        scaledFeatures[:, i] = (features[:, i] - np.min(features[:, i])) / \
            (np.max(features[:, i]) - np.min(features[:, i]))
    return scaledFeatures


def featureScalingUsingSklearn(features):
    '''
        Inputs:
            features: the features to be scaled
        Outputs:
            scaledFeatures: the scaled features
        Note: 
            this function is used to scale the features to be in the range [0,1]
            this is done by subtracting the minimum value from each feature and then dividing by the range
            of the feature.
        Reference: 
            https://stackabuse.com/feature-scaling-data-with-scikit-learn-for-machine-learning-in-python/
    '''
    scaler = StandardScaler()
    scaledFeatures = scaler.fit_transform(features)
    return scaledFeatures


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


def PrincipalComponentAnalysisPCA(NormalizedFeatures):
    '''
        Purpose: 
            * This function is used to extract the most important features from a bulk of features, in order 
            to enhace the accuracy, speed up the model, and get rid of the curse of dimensionality!
            * This is one of the feature extraction techniques.
            * Note that the generated features may have no logical meaning, but they are the best features which 
            affect the performance of the model. 
            * it is an Unsupervised dimensionality reduction technique, where we cluster the similar data points
            based on the features correlation between them without any supervision labels.
        Algorithm:
            * The most interesting directions are those which have the larges variations. 
            * so we move with the following algorithm 
                1. Transform the data to the zero mean: Y = X - U
                2. Estimate covariance matrix from data: C = 1/M Summation(Y(m) * Y^T(m))
                3. Compute the eigenvalues and eigenvectors of the covariance matrix
                4. Sort the eigenvalues in descending order and choose the first K eigenvalues, where K is the
                number of diminsions you want. 
                5. Compute the projection matrix: Z = U^T * Y, where U is the matrix of the eigenvectors
        Inputs:
            NormalizedFeatures: they are the features after applying feature scaling by normalizing them.
        Outputs:
            pcaFeatures: the features after applying PCA.
        Reference:
            lecture 6 notes.
            https://www.datacamp.com/tutorial/principal-component-analysis-in-python
    '''
    ##########################################################################################
    # this part I believe that it is not useful, but I will keep it for now.
    # it just tries to visualise the pca, but we wont use it in the code.
    # #! this n_components should be evaluated dynamically to get the best value for it.
    # pcaFeatureExtractor = PCA(n_components=150000)
    # extractedFeatures = pcaFeatureExtractor.fit_transform(NormalizedFeatures)

    # # ? this will provide us with the amount of information or variance each principal component holds
    # # ? after projecting the data to a lower dimensional subspace.
    # # ? in other words it will tell us each feature carries how much percent of the informations.
    # print('Explained variation per principal component: {}'.format(
    #     extractedFeatures.explained_variance_ratio_))
    ##########################################################################################

    #!!!!!!!!!!!!!!! trying to speed up the code !!!!!!!!!!!!!!!!!!!!#
    # 0.85 tell the PCA how much variance we need and the number of components required to capture 85% variance.
    # Note that earlier you passed n_components as a parameter and you could then
    # find out how much variance was captured by those two components.
    # But here we explicitly mention how much variance we would like PCA to capture and hence,
    # the n_components will vary based on the variance parameter.
    #! this 0.85 also should be modified, I just put it randomly.
    pca = PCA(0.85)

    # fitting pca on the data
    pca.fit(NormalizedFeatures)

    #! this line was existing in the site, but I don't understand whethere I should use it or not
    #  PCA(copy=True, iterated_power='auto', n_components=0.9, random_state=None,
    #   svd_solver='auto', tol=0.0, whiten=False)

    # now we can check on the number of components used
    numberOfComponents = pca.n_components_
    print('number of components: ', numberOfComponents)

    # lets apply the transform function on the data.
    pcaFeatures = pca.transform(NormalizedFeatures)
    return pcaFeatures
#! what is the difference between .fit and .transform?
#! I still need some clarifications.
# https://towardsdatascience.com/fit-vs-transform-in-scikit-libraries-for-machine-learning-3c70e6300ded


def Augmentator(dataSetPath, size):
    '''
        Purpose:
            This function is created mainly as a preprocessing utility function to generate more data
            from the existing data, but with different properties to apply data augmentation and avoid 
            overfitting. 
        Inputs:
            dataSetPath: the path of the dataset
            size: the size of the output augmented data
        Algorithm:
            it takes the path of the dataset, and then apply the following operations
            1. zoom: to zoom in and out the image with certain factors
            2. flip_top_bottom: to flip the image vertically
            3. flip_left_right: to flip the image horizontally
            4. rotate: to rotate the image with certain angles
            5. random_brightness: to change the brightness of the image with certain factors
            6. random_contrast: to change the contrast of the image with certain factors
            7. random_color: to change the color of the image with certain factors
            8. random_distortion: to distort the image with certain factors
        Outputs:
            the augmented data
        Reference:
            https://www.youtube.com/watch?v=mygPhlH8un0 
    '''
    import Augmentor as ag  # if you do not have it -> pip install Augmentor
    # hwa bya5ud el path bta3t el sowar el 3auzen ntb2 3leha el shughl
    p = ag.Pipeline(dataSetPath)

    # applying different operations
    p.zoom(probability=0.2, min_factor=0.5, max_factor=1)
    p.flip_top_bottom(probability=0.5)
    p.flip_left_right(probability=0.5)
    p.rotate(probability=0.7, max_left_rotation=20, max_right_rotation=20)
    p.random_brightness(probability=0.5, min_factor=0.5, max_factor=1.5)
    p.random_contrast(probability=0.5, min_factor=0.5, max_factor=1.5)
    p.random_color(probability=0.5, min_factor=0.5, max_factor=1.5)
    p.random_distortion(probability=0.5, grid_width=4,
                        grid_height=4, magnitude=8)
    return p.sample(size)  # this is the number of images i want to create


def get_pixel(img, center, x, y):
    '''
    Purpose:
        this function is used to get the value of the neighbour pixel, whethere it is 1 or 0
        depending on the LBP algorithm
        if the pixel has higher intensity than the center, then it should be 1, otherwise it should return 0
    Inputs:
        img: our main image
        center: the center of our window
        x: the x coordinate of the pixel
        y: the y cooridnate of the pixel
    Outputs:
        1: if the pixel is greater than the center
        0: otherwise
    Reference: 
        https://www.sciencedirect.com/topics/engineering/local-binary-pattern
    Note: 
        can this code be vectorized? 
    '''
    new_value = 0

    try:
        # If local neighbourhood pixel
        # value is greater than or equal
        # to center pixel values then
        # set it to 1
        if img[x][y] >= center:
            new_value = 1

    except:
        # Exception is required when
        # neighbourhood value of a center
        # pixel value is null i.e. values
        # present at boundaries.
        pass

    return new_value


def lbp_calculated_pixel(img, x, y):
    '''
        Purpose:
            this function is used to evaluate the lbp value of certain pixel. 
        Inputs:
            img: the image on which we apply the lbp
            x: the x coordinate of the center
            y: the y coordinate of the center
        Outputs: 
            the value of this cell in decimal. 
        Reference: 
            https://www.sciencedirect.com/topics/engineering/local-binary-pattern
    '''
    center = img[x][y]

    val_ar = []

    # top_left
    val_ar.append(get_pixel(img, center, x-1, y-1))

    # top
    val_ar.append(get_pixel(img, center, x-1, y))

    # top_right
    val_ar.append(get_pixel(img, center, x-1, y + 1))

    # right
    val_ar.append(get_pixel(img, center, x, y + 1))

    # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))

    # bottom
    val_ar.append(get_pixel(img, center, x + 1, y))

    # bottom_left
    val_ar.append(get_pixel(img, center, x + 1, y-1))

    # left
    val_ar.append(get_pixel(img, center, x, y-1))

    # Now, we need to convert binary
    # values to decimal, assume we work on window of size 3x3.
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]

    val = 0

    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val


def main():
    image = io.imread('hand.jpg', as_gray=True)
    hog_features, hog_image = FeatureExtraction(image)
    io.imshow(hog_features)


if __name__ == '__main__':
    main()
