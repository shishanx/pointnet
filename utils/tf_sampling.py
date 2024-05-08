''' Furthest point sampling
Original author: Haoqiang Fan
Modified by Charles R. Qi
All Rights Reserved. 2017. 
'''
# import tensorflow as tf
# from tensorflow.python.framework import ops
# import sys
import os
import numpy as np
from scipy.spatial import distance
import feature_extraction
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
# sampling_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_sampling_so.so'))
SAMPLE_PATH = 'log/sample'

def log_string(stream, out_str):
    stream.write(out_str+'\n')
    stream.flush()

def the_first_n_visu(points, n_samples, item):
    make_sample_file(points, np.arange(n_samples), np.arange(n_samples, len(points)), item, 'none')

def farthest_point_sample(points, n_samples, item):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N 
    """
    points = np.array(points)
    
    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int') # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf') # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected 
    points_left = np.delete(points_left, selected) # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i-1]
        
        dist_to_last_added_point = (
            (points[last_added] - points[points_left])**2).sum(-1) # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point, 
                                        dists[points_left]) # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    make_sample_file(points, sample_inds, points_left, item, 'fps')

    return points[sample_inds]

def my_point_sample(points, n_samples, item):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N 
    """
    points = np.array(points)
    
    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int') # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf') # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected 
    points_left = np.delete(points_left, selected) # [P - 1]


    # Iteratively select points for a maximum of n_samples
    for i in range(0, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i - 1]
        
        dists[points_left] = ((points[last_added] - points[points_left]) ** 2).sum(-1) # [P - i]

        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    make_sample_file(points, sample_inds, points_left, item, 'mine')

    return points[sample_inds]

def my_point_sample_neighbor(points, n_samples, item, k, points_distance):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N 
    """
    points = np.array(points)
    
    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int') # [S]

    # Select a point from points by its index, save it
    selected = 0

    dist = points_distance
    selected = np.argmax(dist[0])
    sample_inds[0] = selected

    # Delete selected 
    points_left = np.setdiff1d(points_left, [selected])# [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(0, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i - 1]
        
        k_nearest = np.argsort(dist[last_added])
        selected = k_nearest[k - 1]
        sample_inds[i] = selected
        dist[last_added] = np.ones_like(len(points))
        dist[:, last_added] = 1

        # Update points_left
        points_left = np.setdiff1d(points_left, [selected])

    # make_sample_file(points, sample_inds, points_left, item, 'mine_neighbor')

    return points[sample_inds]

def my_point_sample_featured(sess, points, labels, n_samples, item_id, k, points_distance):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N 
    """
    points = np.array(points)
    
    # Represent the points by their indices in points
    # points_left = np.arange(len(points)) # [P]

    # Initialise an array for the sampled indices
    # sample_inds = np.zeros(n_samples, dtype='int') # [S]

    # Select a point from points by its index, save it
    # selected = 0

    dist = points_distance
    neighbor = np.zeros([len(points), k])
    # selected = np.argmax(dist[0])
    # sample_inds[0] = selected

    # Delete selected 
    # points_left = np.setdiff1d(points_left, [selected])# [P - 1]

    for i in range(0, len(points)):
        k_nearest = np.argsort(dist[i])
        neighbor[i] = k_nearest[0 : k]

    sample_point = feature_extraction.featured_extract(sess, points, n_samples, neighbor, item_id)
    if (item_id == "00"):
        make_sample_file(points, sample_point, [], item_id, 'featured')

    return sample_point

def make_sample_file(points, sample_point, left_point, item, method):
    if item != -1 :
        sampled_file = open(os.path.join(SAMPLE_PATH, '%s_%s_%sp_sample.xyzrgb') % (method, item, len(sample_point)), 'w')
        for i in range(0, len(left_point)):
            log_string(sampled_file, '%s %s %s 0 0 1' % (
                points[left_point[i]][0],
                points[left_point[i]][1],
                points[left_point[i]][2]
            ))

        for i in range(0, len(sample_point)):
            log_string(sampled_file, '%s %s %s 1 0 0' % (
                points[sample_point[i]][0],
                points[sample_point[i]][1],
                points[sample_point[i]][2]
            ))