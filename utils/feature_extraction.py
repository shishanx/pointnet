import numpy as np
import tensorflow as tf
import provider
import os
from scipy.spatial import distance
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
tf.placeholder = tf.compat.v1.placeholder
tf.compat.v1.disable_eager_execution()
tf.get_variable = tf.compat.v1.get_variable
tf.Session = tf.compat.v1.Session
tf.global_variables_initializer = tf.compat.v1.global_variables_initializer
tf.truncated_normal_initializer = tf.compat.v1.truncated_normal_initializer

def featured_extract(sess, points, point_num, neighbor, item_id):
    neighbor_num = len(neighbor[0])

    x_in = tf.placeholder("float", [2 * neighbor_num + 1, 3])
    x = tf.reshape(x_in, [1, 2 * neighbor_num + 1, 3, 1])

    # print(x_in.shape)

    conv1_w = tf.get_variable(item_id + "conv1_w", [1, 3, 1, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
    net = tf.nn.conv2d(x, conv1_w, [1, 1, 1, 1], "VALID")
    # print(net.shape)

    conv2_w = tf.get_variable(item_id + "conv2_w", [1, 1, 64, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
    net = tf.nn.conv2d(net, conv2_w, [1, 1, 1, 1], "VALID")
    # print(net.shape)

    conv3_w = tf.get_variable(item_id + "conv3_w", [1, 1, 64, 64], initializer=tf.truncated_normal_initializer(stddev=0.1))
    net = tf.nn.conv2d(net, conv3_w, [1, 1, 1, 1], "VALID")
    # print(net.shape)

    conv4_w = tf.get_variable(item_id + "conv4_w", [1, 1, 64, 128], initializer=tf.truncated_normal_initializer(stddev=0.1))
    net = tf.nn.conv2d(net, conv4_w, [1, 1, 1, 1], "VALID")
    # print(net.shape)

    conv5_w = tf.get_variable(item_id + "conv5_w", [1, 1, 128, 1024], initializer=tf.truncated_normal_initializer(stddev=0.1))
    net = tf.nn.conv2d(net, conv5_w, [1, 1, 1, 1], "VALID")
    # print(net.shape)

    net = tf.nn.max_pool(net, [1, 17, 1, 1], [1, 1, 1, 1], padding='VALID')
    # print(net.shape)

    net = tf.reshape(net, [1, -1])
    # print(net.shape)

    fc1_w = tf.get_variable(item_id + "fc1_w", [1024, 512], initializer=tf.truncated_normal_initializer(stddev=0.1))
    net = tf.matmul(net, fc1_w)
    # print(net.shape)

    net = tf.nn.dropout(net, 0.7)
    fc2_w = tf.get_variable(item_id + "fc2_w", [512, 256], initializer=tf.truncated_normal_initializer(stddev=0.1))
    net = tf.matmul(net, fc2_w)
    # print(net.shape)

    net = tf.nn.dropout(net, 0.7)
    fc3_w = tf.get_variable(item_id + "fc3_w", [256, 3], initializer=tf.truncated_normal_initializer(stddev=0.1))
    net = tf.matmul(net, fc3_w)

    # print(net)


    init = tf.global_variables_initializer()
    # current_data, current_label = provider.loadDataFile(TRAIN_FILES[0])
    # point_num = 128
    result = np.zeros([point_num, 3])
    mlp_input = np.zeros([2 * neighbor_num + 1, 3], dtype=np.float32)

    for index in range(1):
        item = points
        # # item中每個點跟其他點的距離
        # dist = distance.squareform(distance.pdist(item))
        # # 存item中每個點的neighbor
        # neighbor = np.zeros([len(item), neighbor_num])

        for i in range(point_num):
            # 中心點i
            mlp_input[0] = item[i]
            # # 排序i對每個點的距離
            # k_nearest = np.argsort(dist[i])
            # # i的前幾個鄰居點
            # neighbor[i] = k_nearest[0 : neighbor_num]
            for j in range(1, neighbor_num):
                mlp_input[j] = item[int(neighbor[i][j])]
                mlp_input[j + neighbor_num] = item[i] - item[int(neighbor[i][j])]
            # xtest = np.arange(51)
            # xtest = np.reshape(xtest, [17, 3])
            xtest = mlp_input

            sess.run(init)
            result[i] = sess.run(net, feed_dict={x_in: xtest})[0]

    # print(result)
    print(item_id)
    return result