import random
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from skimage import io, measure, exposure
import h5py


# This is a function that reads a image list from data directory.
def image_list(datadir):
    """
    This function takes directory that contains sub categories of images and return a shuffled list of images.
    """
    categories = os.listdir(datadir)
    f"Listing images from: {categories}"
    image_list = []
    for category in categories:
        path = os.path.join(datadir, category)
        imgs = [img for img in os.listdir(path) if img.endswith(".tif")]
        image_list.extend(imgs)
    shuffled_list = random.sample(image_list, len(image_list))
    return shuffled_list


def read_and_process_image(datadir):
    """
    inputs : directory of the data.
    Returns two arrays:
        X is an array of images
        y is an array of labels
    """
    img_arrays = []  # images
    labels = []  # labels
    image_list = []
    categories = os.listdir(datadir)
    print(" Reading images from: {}".format(categories))
    for category in categories:
        path = os.path.join(datadir, category)
        imgs = [img for img in os.listdir(path) if img.endswith(".tif")]
        image_list.extend(imgs)

    shuffled_list = random.sample(image_list, len(image_list))
    path_a = os.path.join(datadir, A)
    path_v = os.path.join(datadir, V)
    for image in shuffled_list:
        if '112A' in image:
            img = io.imread(os.path.join(path_a, image), as_gray=True)  # Read the image
            scaler = MinMaxScaler(copy=True)
            scaler.fit(img)
            scaled_img = scaler.transform(img)  # normalizing the image
            equalized_hist = exposure.equalize_hist(scaled_img)

            img_arrays.append(equalized_hist)
        else:
            img = io.imread(os.path.join(path_v, image), as_gray=True)
            scaler = MinMaxScaler(copy=True)
            scaler.fit(img)
            scaled_img = scaler.transform(img)  # normalizing the image
            equalized_hist = exposure.equalize_hist(scaled_img)
            img_arrays.append(equalized_hist)
        # get the labels
        if '112A' in image:
            labels.append(0)
        elif '112V' in image:
            labels.append(1)

    return img_arrays, labels


def scaller_finder(datadir):
    """
        This function is useful to figuring out what scaller value to use for doing
        Min-Max scaller.
        Args:
        datadir : Directory of the images. This should be the absolute path.
        Put the input in ' '. eg: '/capstoneimage/A/'
        Returns:
        minimum and maximum pixel value of the image stacks.
        """
    categories = ['V', 'A']
    img_arrays = []
    for category in categories:
        stack = read_3d_volume(os.path.join(datadir, category))
        img_arrays.extend(stack)
    images = np.asarray(img_arrays)
    my_max = images.max()
    my_min = images.min()
    print(" The min pixel value of all the images is {}.\
      The max pixel value of all the images is {},".format(my_min, my_max))
    return [my_min, my_max]


def read_3d_volume(datadir):
    """This function uses scikit image imread function to read stack of images.
    normalize them using the function "normalized_X = (X - min)/ (max-min)".
    Images are normalize using the same scaling factor, because all the images
    in given directory is normalized using the min and max values of all the
    pixels.
    Args:
      datadir: Directory of the images. This should be the absolute path.
    Returns:
      3D numpy array with shape (number_of_images_in_the_directory, img_size, img_size).
    """
    img_arrays = []
    for img in os.listdir(datadir):
        if img.endswith(".tif"):
            image = io.imread(os.path.join(datadir, img), as_gray=True)  # Read the image
            img_arrays.append(image)
    stack = np.array(img_arrays)
    # my_max = stack.max()
    # my_min = stack.min()
    # scalled_stack = (stack - my_min) / (my_max-my_min)
    return stack


def rand_xyz_box(image_arrays, label, n, depth, img_size):
    """Returns n number of randomly chosen box.
    
     Args:
        image_arrays: 3D np array of images.
        label: label of images. normally is A or V
        n: number of random boxes generated from this function.
        depth : number of slices in Z direction. default is 50 if not specified.
        img_size: image size in X,Y directions. default is 50 if not specified.
    Returns:
        List object. ['Z','X','Y','im_array','labels'].
        Each im_array is a randomly chosen box with volume of depth*img_size*img_size.
    """
    z = np.random.randint(len(image_arrays)-depth+1, size=n)
    x = np.random.randint(len(image_arrays[1])-img_size+1, size=n)
    y = np.random.randint(len(image_arrays[2])-img_size+1, size=n)
    n_box = []
    for z, x, y in zip(z, x, y):
        box = image_arrays[z:z+depth, x:x+img_size, y:y+img_size]
        box = np.reshape(box, (depth, img_size, img_size, 1))
        n_box.append([z, x, y, box, label])
    return n_box


def slide_xy_box(image_arrays, label, n, depth, img_size):
    """Returns n number of volume from image_arrays. x and y 
    are chosen to make sure there is no overlapping in choosen volume
    at given z. z is chosen randomly with 20 in between 2 z value.
    Efforts are made to minimize the cross_over of choosen volume.
    
     Args:
        image_arrays: 3D np array of images.
        label: label of images. normally is A or V
        n: number of random boxes generated from this function.
        depth : number of slices in Z direction. default is 50 if not specified.
        img_size: image size in X,Y directions. default is 50 if not specified.
    Returns:
        List object. ['Z','X','Y','im_array','labels'].
        Each im_array is a randomly chosen box with volume (depth*img_size*img_size)
    """
    n_box = []
    num = n * (int(round(len(image_arrays[2])/img_size))**2)
    size = '{}_{}_{}'.format(depth, img_size, img_size)
    print(" Creating {} box from this image stack, with size {}".format(num, size))
    for z in range(n):
        z = random.randrange(0, len(image_arrays)-depth+1, 20)
        for x in range(int(round(len(image_arrays[1])/img_size))):
            for y in range(int(round(len(image_arrays[2])/img_size))):
                box = image_arrays[z:z+depth, x*img_size:x*img_size+img_size,
                                   y*img_size:y*img_size+img_size]
                box = np.reshape(box, (depth, img_size, img_size, 1))
                n_box.append([z, x*img_size, y*img_size, box, label])
    return n_box


def min_overlap_3d_data(datadir, exporting_path, n, depth=25, img_size=50):
    """
    This This function uses scikit image imread function to read stack of images.
    normalize them using the function "normalized_X = (X - min)/ (max-min)".
    Images are normalize using the same scaling factor, because all the images
    in given directory is normalized using the min and max values of all the
    pixels.
    Args:
        datadir: Directory of the images. This should be the absolute path.
        exporting_path: directory that you want to store your output h5 data file.
        n: number of random z values for generating image volume.
        depth : number of slices in Z direction. default is 50 if not specified.
        img_size: image size in X,Y directions. default is 50 if not specified.
    Returns:
        Saved h5 file to the exporting path.
    """
    categories = ['V', 'A']
    print(" Reading images from directory {}, has two\
          sub categories {}".format(datadir, categories))
    data = []
    for category in categories:
        print(" Reading {} images.".format(category))
        img_arrays = read_3d_volume(os.path.join(datadir, category))
        print(" Finish reading {} images. It has {} images.".format(category, len(img_arrays)))
        scaller = scaller_finder(datadir)
        scalled_images = (img_arrays - scaller[0]) / (scaller[1] - scaller[0])
        box = slide_xy_box(scalled_images, category, n, depth=depth, img_size=img_size)
        data.extend(box)   
    random.shuffle(data)
    print('Finished creating volume data. Now saving it into hdf5 file format') 
    img_data = np.array([data[i][3] for i in range(len(data))])
    label = np.array([data[i][4] for i in range(len(data))])
    transfer_label = [np.string_(i) for i in label]
    location_data = [[data[i][0], data[i][1], data[i][2]]for i in range(len(data))]
    name = 'no_{}_{}_{}_{}.h5'.format(n, depth, img_size, img_size)
    path = os.path.join(exporting_path, name)
    print(" Saving file with name {}, at path {}".format(name, exporting_path))
    with h5py.File(path, 'w') as f:
        f.create_dataset('slice_location', data=location_data)
        f.create_dataset('img_data', data=img_data)
        f.create_dataset('labels', data=transfer_label)
        f.close()
    return
    

def random_3d_data(datadir, exporting_path, n, depth=25, img_size=50):
    """
    This This function uses scikit image imread function to read stack of images.
    normalize them using the function "normalized_X = (X - min)/ (max-min)".
    Images are normalize using the same scaling factor, because all the images
    in given directory is normalized using the min and max values of all the
    pixels.
    Args:
        datadir: Directory of the images. This should be the absolute path.
        exporting_path: directory that you want to store your output h5 data file.
        n: number of random boxes generated from this function.
        depth : number of slices in Z direction. default is 50 if not specified.
        img_size: image size in X,Y directions. default is 50 if not specified.
    Returns:
    Saved h5 file to the exporting path.
    """
    categories = ['V', 'A']
    print(" Reading images from directory {}, has two\
          sub categories {}".format(datadir, categories))
    data = []
    for category in categories:
        print(" Reading {} images.".format(category))
        img_arrays = read_3d_volume(os.path.join(datadir, category))
        print(" Finish reading {} images. It has {} images.".format(category, len(img_arrays)))
        scaller = scaller_finder(datadir)
        scalled_images = (img_arrays - scaller[0]) / (scaller[1] - scaller[0])
        print(" Creating {} randomly chosen image volumes.".format(n))
        box = rand_xyz_box(scalled_images, category, n, depth=depth, img_size=img_size)
        data.extend(box)   
    random.shuffle(data)
    print('Finished creating volume data. Now saving it into hdf5 file format') 
    img_data = np.array([data[i][3] for i in range(len(data))])
    label = np.array([data[i][4] for i in range(len(data))])
    transfer_label = [np.string_(i) for i in label]
    location_data = [[data[i][0], data[i][1], data[i][2]]for i in range(len(data))]
    name = 'rand_{}_{}_{}_{}.h5'.format(n, depth, img_size, img_size)
    path = os.path.join(exporting_path, name)
    print(" Saving file with name {}, at path {}".format(name, exporting_path))
    with h5py.File(path, 'w') as f:
        f.create_dataset('slice_location', data=location_data)
        f.create_dataset('img_data', data=img_data)
        f.create_dataset('labels', data=transfer_label)
        f.close()
    return
