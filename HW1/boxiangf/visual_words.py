import os, multiprocessing
from os.path import join, isfile

import numpy as np
from PIL import Image
import scipy.ndimage
import scipy.spatial
import skimage.color
import sklearn.cluster

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales
    # ----- TODO -----

    # Convert gray-scale images to 3-channel
    if len(img.shape) != 3:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    # Convert image to floating point type with range [0, 1] if necessary
    if img.dtype.kind != "f":
        img = np.array(img).astype(np.float32)

    if np.any((img < 0) | (img > 1)):
        img = (img-np.min(img))/(np.max(img)-np.min(img))

    img = skimage.color.rgb2lab(img)
    
    # Q3.2 Subtract mean color in each channel. Comment out if not Q3.2
    # r_avg, g_avg, b_avg = np.mean(img, axis=(0,1))
    # img[:,:,0] -= r_avg
    # img[:,:,1] -= g_avg
    # img[:,:,2] -= b_avg

    num_scales = len(filter_scales)
    num_filters = 4
    num_channels = 3

    filter_responses = np.zeros((img.shape[0], img.shape[1], num_channels*num_scales*num_filters))

    # Grouping filter_responses first by channel, then by filter, and finally by scale
    for i in range(num_scales):
        for j in range(num_channels):
            filter_responses[:, :, num_filters*num_channels*i + num_channels*0 + j] = scipy.ndimage.gaussian_filter(img[:, :, j], sigma=filter_scales[i], order=0)
            filter_responses[:, :, num_filters*num_channels*i + num_channels*1 + j] = scipy.ndimage.gaussian_laplace(img[:, :, j], sigma=filter_scales[i])
            filter_responses[:, :, num_filters*num_channels*i + num_channels*2 + j] = scipy.ndimage.gaussian_filter(img[:, :, j], sigma=filter_scales[i], order=[1,0])
            filter_responses[:, :, num_filters*num_channels*i + num_channels*3 + j] = scipy.ndimage.gaussian_filter(img[:, :, j], sigma=filter_scales[i], order=[0,1])
    
    return filter_responses


def compute_dictionary_one_image(args):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''

    # ----- TODO -----
    
    index = args[0]
    alpha = args[1]
    file_name = args[2]
    opts = args[3]
    
    img_path = join(opts.data_dir, file_name)
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    
    filter_responses = extract_filter_responses(opts, img)
    
    # Collect a subset of alpha pixels randomly and save it to a temp file
    x_values = np.random.choice(filter_responses.shape[0], size=alpha, replace=True)
    y_values = np.random.choice(filter_responses.shape[1], size=alpha, replace=True)
    subset_filter_responses = filter_responses[x_values, y_values, :]
    
    np.save(join(opts.data_dir + '/temp/' + str(index)), subset_filter_responses)


def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    # ----- TODO -----
    
    num_training_images = len(train_files)
    alpha = opts.alpha
    
    # Multiprocessing pool for extracting features of one image
    args = []
    for i in range(num_training_images):
        args.append((i, alpha, train_files[i], opts))
    
    pool = multiprocessing.Pool(n_worker)
    pool.map(compute_dictionary_one_image, args)
    pool.close()
    pool.join()
    
    # Read the subset filter responses from file and concatenate
    filter_responses_list = []
    for i in range(num_training_images):
        file_path = join(opts.data_dir + '/temp/' + str(i) + ".npy")
        file = np.load(file_path)
        filter_responses_list.append(file)
    filter_responses = np.concatenate(filter_responses_list, axis=0)
    
    # Cluster responses with k-means and generate visual words dictionary with K words
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(filter_responses)
    dictionary = kmeans.cluster_centers_
    np.save(join(out_dir, 'dictionary.npy'), dictionary)


def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    
    img_height = img.shape[0]
    img_width = img.shape[1]
    
    # Collect filter responses and reshape to a 2-dimensional matrix of size (img_height*img_width, 3*filter_bank_size)
    filter_responses = extract_filter_responses(opts, img)
    filter_responses_2d = np.reshape(filter_responses, (img_height*img_width, filter_responses.shape[2]))
    
    # Calculate Euclidean distance to each vector in dictionary and find the closest visual word 
    euclidean_distance = scipy.spatial.distance.cdist(filter_responses_2d, dictionary, metric='euclidean')
    closest_visual_word_index = np.argmin(euclidean_distance, axis=1)
    wordmap = np.reshape(closest_visual_word_index, (img_height, img_width))

    return wordmap