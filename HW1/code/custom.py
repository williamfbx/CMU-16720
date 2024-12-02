from os.path import join

import numpy as np
import itertools as it

import visual_words
import visual_recog
from opts import custom_get_opts

def tune_hyperparams(opts, n_worker=1):
    '''
    Enumerate a list of potential hyperparameters and find the best one

    [input]
    * opts: options
    * n_worker: number of workers to process in parallel

    [output]
    * optimal_params_list: list of optimal hyperparameters for the model
    '''
    
    default_filter_scales = opts.filter_scales
    default_K = opts.K
    default_alpha = opts.alpha
    default_L = opts.L
    default_accuracy = 0
    
    optimal_filter_scales = default_filter_scales
    optimal_K = default_K
    optimal_alpha = default_alpha
    optimal_L = default_L
    optimal_accuracy = default_accuracy
    optimal_conf = np.zeros((8, 8))
    
    potential_filter_scales = [[1, 2], [1, 2, 4], [1, 2, 4, 8]]
    potential_K = [10, 20, 40, 80, 160]
    potential_alpha = [25, 50, 100, 200]
    potential_L = [1, 2, 3]
    
    hyperparam_combinations = it.product(potential_filter_scales, potential_K, potential_alpha, potential_L)
    
    for combination in hyperparam_combinations:
        
        curr_filter_scale = combination[0]
        curr_K = combination[1]
        curr_alpha = combination[2]
        curr_L = combination[3]
        
        # Populate parameters and run the model on these parameters
        custom_opts = custom_get_opts(curr_filter_scale, curr_K, curr_alpha, curr_L)
        
        visual_words.compute_dictionary(custom_opts, n_worker=n_worker)
        visual_recog.build_recognition_system(custom_opts, n_worker=n_worker)
        conf, accuracy = visual_recog.evaluate_recognition_system(custom_opts, n_worker=n_worker)
        
        if accuracy > optimal_accuracy:
            optimal_accuracy = accuracy
            optimal_filter_scales = curr_filter_scale
            optimal_K = curr_K
            optimal_alpha = curr_alpha
            optimal_L = curr_L
            optimal_conf = conf
        
        print("Running on hyperparameters: " + "filter-scale: " + str(curr_filter_scale) + ", K: " + str(curr_K) + ", alpha: " + str(curr_alpha) + ", L: " + str(curr_L) + " achieved an accuracy of " + str(accuracy))
    
    # Save the optimal accuracy and confusion matrix to file
    np.savetxt(join(opts.out_dir, 'optimal_confmat.csv'), optimal_conf, fmt='%d', delimiter=',')
    np.savetxt(join(opts.out_dir, 'optimal_accuracy.txt'), [optimal_accuracy], fmt='%g')
    
    return (optimal_filter_scales, optimal_K, optimal_alpha, optimal_L)
