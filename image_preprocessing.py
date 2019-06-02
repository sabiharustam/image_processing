import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from skimage import io, measure, exposure

# This is a function that reads a image list from data directory.
def image_list (DATADIR):
    """
    This function takes directory that contains sub categories of images and return a shuffled list of images.
    """
    CATEGORIES = os.listdir(DATADIR)
    f"Listing images from: {CATEGORIES}"
    image_list = []
    for categories in CATEGORIES:
        path = os.path.join(DATADIR, categories)
        imgs = [img for img in os.listdir(path) if img.endswith(".tif")]
        image_list.extend(imgs)
    #print(image_list)
    shuffled_list= random.sample(image_list,len(image_list))
    # random.shuffle returns none instead.
    #print(shuffled_list)
    return(shuffled_list)


def read_and_process_image(DATADIR):
    """
    inputs : directory of the data.
    Returns two arrays:
        X is an array of images
        y is an array of labels
    """
    X = []  # images
    y = []  # labels
    image_list = []
    CATEGORIES = os.listdir(DATADIR)
    print(" Reading images from: {}".format(CATEGORIES))
    for categories in CATEGORIES:
        path = os.path.join(DATADIR, categories)
        imgs = [img for img in os.listdir(path) if img.endswith(".tif")]
        image_list.extend(imgs)

    shuffled_list = random.sample(image_list, len(image_list))
    path_A = os.path.join(DATADIR, CATEGORIES[0])
    path_V = os.path.join(DATADIR, CATEGORIES[1])
    for image in shuffled_list:
        if '112A' in image:
            img = io.imread(os.path.join(path_A, image), as_gray=True)  # Read the image
            scaler = MinMaxScaler(copy=True)
            scaler.fit(img)
            scaled_img = scaler.transform(img)  # normalizing the image
            equalized_hist = exposure.equalize_hist(scaled_img)
            # new_array = skimage.transform.resize(equalized_hist,[nrows,ncolumns])
            X.append(equalized_hist)
        else:
            img = io.imread(os.path.join(path_V, image), as_gray=True)
            scaler = MinMaxScaler(copy=True)
            scaler.fit(img)
            scaled_img = scaler.transform(img)  # normalizing the image
            equalized_hist = exposure.equalize_hist(scaled_img)
            # new_array = skimage.transform.resize(equalized_hist,[nrows,ncolumns])
            X.append(equalized_hist)
        # get the labels
        if '112A' in image:
            y.append(0)
        elif '112V' in image:
            y.append(1)

    return X, y


def read_3D_volume(DATADIR):
    X = []
    y = []
    for img in os.listdir(DATADIR):
        if img.endswith(".tif"):
            image = io.imread(os.path.join(DATADIR,img),as_gray=True) #Read the image
            scaler = MinMaxScaler(copy=True)
            scaler.fit(image)
            scaled_img = scaler.transform(image) # normalizing the image
            equalized_hist = exposure.equalize_hist(scaled_img)
            X.append(equalized_hist)
        if '112A' in img:
            y.append(0)
        elif '112V' in img:
            y.append(1)
    X = np.array(X)
    y = np.array(y)
    return X,y


# If we want n number of random box from Volume A or V:
def N_sliced_box(image_arrays,label, n, SLICE_NUM = 25, IMG_SIZE = 50,):
    """
    Inputs:
    -------
    image_arrays: 3D np array of images.
    n: number of random boxs generated from this function.
    SLICE_NUM : number of slices in Z direction. default is 50 if not specified.
    IMG_SIZE: image size in X,Y directions. default is 50 if not specified.
    
    Returns:
    --------
    Pandas DataFrame. columns=['Z','X','Y','im_array','labels'].
    Each im_array is a randomly choosen box with volume of SLICE_NUM*IMG_SIZE*IMG_SIZE.
    """
    #n_box = []
    z = np.random.randint(len(image_arrays[0])-SLICE_NUM+1, size= n)
    x = np.random.randint(len(image_arrays[1])-IMG_SIZE+1, size= n)
    y = np.random.randint(len(image_arrays[2])-IMG_SIZE+1, size= n)
    df = pd.DataFrame(columns=['Z','X','Y','im_array','labels'])
    for z,x,y in zip(z,x,y):
        box = image_arrays[z:z+SLICE_NUM,x:x+IMG_SIZE,y:y+IMG_SIZE]
        df = df.append(pd.Series([z, x, y,box,label], index=['Z','X','Y','im_array','labels']), ignore_index=True)
        #print(" Created volume from z= {}, x = {}, y= {}".format(z,x,y))
        #n_box.append(box)
    return df