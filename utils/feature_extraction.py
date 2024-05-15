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

def featured_extract(sess, epoch, points, point_num, k):
    neighbor_num = k
    item_num = len(points)
    item_point_num = len(points[0])
    x_in = tf.placeholder("float", [item_num, item_point_num, 2 * neighbor_num + 1, 3])
    # x = tf.reshape(x_in, [item_num, 2 * neighbor_num + 1, 3, 1])

    # print(x_in.shape)

    conv1_w = tf.get_variable(str(epoch) + "_conv1_w", [1, 2 * k + 1, 3, 64], initializer=tf.compat.v1.keras.initializers.glorot_normal())
    net = tf.nn.conv2d(x_in, conv1_w, [1, 1, 1, 1], "VALID")
    net = tf_util.batch_norm_for_conv2d(net, tf.constant(True), bn_decay=True, scope=str(epoch) + 'bn1')
    net = tf.nn.relu(net)
    # print(net.shape)

    # conv2_w = tf.get_variable(str(epoch) + "_conv2_w", [1, 1, 64, 64], initializer=tf.compat.v1.keras.initializers.glorot_normal())
    # net = tf.nn.conv2d(net, conv2_w, [1, 1, 1, 1], "VALID")
    # net = tf_util.batch_norm_for_conv2d(net, tf.constant(True), bn_decay=True, scope=str(epoch) + 'bn2')
    # net = tf.nn.relu(net)
    # # print(net.shape)

    # conv3_w = tf.get_variable(str(epoch) + "_conv3_w", [1, 1, 64, 64], initializer=tf.compat.v1.keras.initializers.glorot_normal())
    # net = tf.nn.conv2d(net, conv3_w, [1, 1, 1, 1], "VALID")
    # net = tf_util.batch_norm_for_conv2d(net, tf.constant(True), bn_decay=True, scope=str(epoch) + 'bn3')
    # net = tf.nn.relu(net)
    # # print(net.shape)

    conv4_w = tf.get_variable(str(epoch) + "_conv4_w", [1, 1, 64, 128], initializer=tf.compat.v1.keras.initializers.glorot_normal())
    net = tf.nn.conv2d(net, conv4_w, [1, 1, 1, 1], "VALID")
    net = tf_util.batch_norm_for_conv2d(net, tf.constant(True), bn_decay=True, scope=str(epoch) + 'bn4')
    net = tf.nn.relu(net)
    # print(net.shape)

    conv5_w = tf.get_variable(str(epoch) + "_conv5_w", [1, 1, 128, 1024], initializer=tf.compat.v1.keras.initializers.glorot_normal())
    net = tf.nn.conv2d(net, conv5_w, [1, 1, 1, 1], "VALID")
    net = tf_util.batch_norm_for_conv2d(net, tf.constant(True), bn_decay=True, scope=str(epoch) + 'bn5')
    net = tf.nn.relu(net)
    # print(net.shape)

    max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2 * k + 1, 1), strides=(1, 1), padding="VALID", data_format="channels_last")
    max_pool_2d(net)
    # net = tf.nn.max_pool(net, [item_num, 2 * k + 1, 1, 1], [1, 1, 1, 1], padding='VALID')
    # print(net.shape)

    net = tf.reshape(net, [item_num, item_point_num, -1])
    # print(net.shape)

    fc1_w = tf.get_variable(str(epoch) + "_fc1_w", [1024, 512], initializer=tf.compat.v1.keras.initializers.glorot_normal())
    net = tf.matmul(net, fc1_w)
    net = tf_util.batch_norm_for_fc(net, tf.constant(True), bn_decay=True, scope=str(epoch) + 'bn6')
    net = tf.nn.relu(net)
    # print(net.shape)

    net = tf.nn.dropout(net, 0.7)
    fc2_w = tf.get_variable(str(epoch) + "_fc2_w", [512, 256], initializer=tf.compat.v1.keras.initializers.glorot_normal())
    net = tf.matmul(net, fc2_w)
    net = tf_util.batch_norm_for_fc(net, tf.constant(True), bn_decay=True, scope=str(epoch) + 'bn7')
    net = tf.nn.relu(net)
    # print(net.shape)

    net = tf.nn.dropout(net, 0.7)
    fc3_w = tf.get_variable(str(epoch) + "_fc3_w", [256, 1], initializer=tf.compat.v1.keras.initializers.glorot_normal())
    net = tf.matmul(net, fc3_w)
    net = tf_util.batch_norm_for_fc(net, tf.constant(True), bn_decay=True, scope=str(epoch) + 'bn8')
    net = tf.nn.relu(net)

    # print(net)


    init = tf.global_variables_initializer()
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

    sess.run(init)
    net_result = sess.run(net, feed_dict={x_in: xtest})
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