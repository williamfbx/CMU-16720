import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import visual_words


def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K
    # ----- TODO -----
    
    wordmap_1d = np.reshape(wordmap, wordmap.shape[0]*wordmap.shape[1])
    bin_edges = [i for i in range(K+1)]
    
    # Histogram normalized to L1 norm
    (hist_tally, edges_tally) = np.histogram(wordmap_1d, bins=bin_edges)
    hist = hist_tally / np.sum(hist_tally)
    
    return hist
    

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L
    # ----- TODO -----
    
    img_histograms = []
    
    for layer in range(L, -1, -1):
        img_partitions = []
        num_partitions = pow(2, layer)
        
        # Calculate weight of layer
        if (layer == 0) or (layer == 1):
            weight = np.float_power(2, -L)
        else:
            weight = np.float_power(2, layer-L-1)
            
        # Partition image into num_partition * num_partition grids
        img_partition_one_axis = np.array_split(wordmap, num_partitions, axis=1)
        for i in range(num_partitions):
            img_partition_two_axis = np.array_split(img_partition_one_axis[i], num_partitions, axis=0)
            for j in range(num_partitions):
                img_partitions.append(img_partition_two_axis[j])
            
        # Scale histograms by weight and divide by number of partition grids
        for img in img_partitions:
            scaling_factor = weight * 1/(num_partitions*num_partitions)
            img_histograms.append(scaling_factor * get_feature_from_wordmap(opts, img))
    
    # Correct for minor floating-point errors
    hist_all = np.hstack(img_histograms)
    hist_all = hist_all / sum(hist_all)
    
    return hist_all
    
    
def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K*(4^L-1)/3)
    '''

    # ----- TODO -----
    
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts, wordmap)
    
    return feature


def get_image_feature_parallel(args):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [saved]
    * feature: numpy.ndarray of shape (M)
    '''

    # ----- TODO -----
    
    index = args[0]
    opts = args[1]
    img_path = args[2]
    dictionary = args[3]
    
    img = Image.open(img_path)
    img = np.array(img).astype(np.float32)/255
    
    wordmap = visual_words.get_visual_words(opts, img, dictionary)
    feature = get_feature_from_wordmap_SPM(opts, wordmap)
    
    np.save(join(opts.data_dir + '/temp/' + str(index)), feature)


def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))

    # ----- TODO -----
    
    # No parallel computing version
    # features_list = []
    
    # for file_name in train_files:
    #     img_path = join(opts.data_dir, file_name)
    #     img_feature = get_image_feature(opts, img_path, dictionary)
    #     features_list.append(img_feature)
    
    # features = np.vstack(features_list)
    
    # Parallel computing version
    num_files = len(train_files)
    args = []

    # Multiprocessing pool for extracting features of all images
    for i in range(num_files):
        file_name = train_files[i]
        img_path = join(opts.data_dir, file_name)
        args.append((i, opts, img_path, dictionary))
    
    pool = multiprocessing.Pool(n_worker)
    pool.map(get_image_feature_parallel, args)
    pool.close()
    pool.join()

    # Read each feature from file
    features_list = []
    for i in range(num_files):
        file_path = join(opts.data_dir + '/temp/' + str(i) + ".npy")
        file = np.load(file_path)
        features_list.append(file)
    features = np.vstack(features_list)
        
    ## example code snippet to save the learned system
    np.savez_compressed(join(out_dir, 'trained_system.npz'),
        features=features,
        labels=train_labels,
        dictionary=dictionary,
        SPM_layer_num=SPM_layer_num,
    )


def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * hist_dist: numpy.ndarray of shape (N)
    '''

    # ----- TODO -----
    hist_min_intersect = np.minimum(word_hist, histograms)
    hist_sim = np.sum(hist_min_intersect, axis=1)
    hist_dist = 1 - hist_sim
    
    return hist_dist
    
    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']

    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']

    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)

    # ----- TODO -----
    
    trained_features = trained_system['features']
    trained_labels = trained_system['labels']

    conf = np.zeros((8, 8))
    
    num_test_files = len(test_files)
    
    # Assigns predicted label based on the label of the closest SPM vector in the trained dataset
    for i in range(num_test_files):
        test_file_name = test_files[i]
        img_path = join(opts.data_dir, test_file_name)
        img_feature = get_image_feature(test_opts, img_path, dictionary)
        distance_to_trained_sample = distance_to_set(img_feature, trained_features)
        closest_trained_sample_index = np.argmin(distance_to_trained_sample)
        
        true_label = test_labels[i]
        predicted_label = trained_labels[closest_trained_sample_index]
        conf[true_label, predicted_label] += 1
        
    # Calculate accuracy of the classification
    accuracy = np.trace(conf) / np.sum(conf)
    
    return conf, accuracy