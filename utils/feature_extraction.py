import numpy as np
import tensorflow as tf
import provider
import os
from scipy.spatial import distance
import tf_util
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
tf.placeholder = tf.compat.v1.placeholder
tf.compat.v1.disable_eager_execution()
tf.get_variable = tf.compat.v1.get_variable
tf.Session = tf.compat.v1.Session
tf.global_variables_initializer = tf.compat.v1.global_variables_initializer
tf.truncated_normal_initializer = tf.compat.v1.truncated_normal_initializer

def featured_extract(sess, ops, epoch, points, point_num, k):
    neighbor_num = k
    item_num = len(points)
    item_point_num = len(points[0])

    # print(net)


    # current_data, current_label = provider.loadDataFile(TRAIN_FILES[0])
    # point_num = 128
    result = np.zeros([item_num, point_num, 3])

    mlp_input = np.zeros([item_num, item_point_num, 2 * neighbor_num + 1, 3], dtype=np.float32)

    for index in range(item_num):
        item = points[index]
        dist = distance.squareform(distance.pdist(item))
        neighbor = np.zeros([len(item), k])
        print(str(epoch) + " item: " + str(index), end="\r")

        for i in range(len(item)):
            k_nearest = np.argsort(dist[i])
            neighbor[i] = k_nearest[0 : k]
            # 中心點i
            mlp_input[index][i][0] = item[i]
            # # 排序i對每個點的距離
            # k_nearest = np.argsort(dist[i])
            # # i的前幾個鄰居點
            # neighbor[i] = k_nearest[0 : neighbor_num]
            for j in range(1, neighbor_num):
                mlp_input[index][i][j] = item[int(neighbor[i][j])]
                mlp_input[index][i][j + neighbor_num] = item[i] - item[int(neighbor[i][j])]
            # xtest = np.arange(51)
            # xtest = np.reshape(xtest, [17, 3])
    
    xtest = mlp_input
    net_result = sess.run(ops['net'], feed_dict={ops['x_in']: xtest})
    net_result = np.reshape(net_result, [item_num, item_point_num])

    result = np.zeros([item_num, point_num, 3])
    result_index = np.zeros([item_num, point_num])
    for i in range(len(net_result)):
        score_sort = np.flip(np.argsort(net_result[i]))
        first_n_score = score_sort[0 : point_num]
        result_index[i] = first_n_score
        for j in range(len(first_n_score)):
            result[i][j] = points[i][first_n_score[j]]
    # print(item_id)
    return result, result_index