import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pc_util
import imageio
import tf_util
from tf_sampling import farthest_point_sample, my_point_sample, my_point_sample_neighbor, my_point_sample_featured
from scipy.spatial import distance
from transform_nets import input_transform_net, feature_transform_net
tf.train = tf.compat.v1.train
tf.ConfigProto = tf.compat.v1.ConfigProto
tf.global_variables = tf.compat.v1.global_variables
tf.global_variables_initializer = tf.compat.v1.global_variables_initializer
tf.Session = tf.compat.v1.Session


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
parser.add_argument('--visu', action='store_true', help='Whether to dump image for error case [default: False]')
parser.add_argument('--sample', type=str, default='none', help='Sampling')
parser.add_argument('--sample_batch_num', type=int, default=256, help='Batch Num during sampling [default: 256]')
parser.add_argument('--sample_path', default='log/sample.ckpt', help='model checkpoint file path [default: log/sample.ckpt]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
SAMPLING = FLAGS.sample
SAMPLE_BATCH_NUM = FLAGS.sample_batch_num
SAMPLE_PATH = FLAGS.sample_path
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 40
SHAPE_NAMES = [line.rstrip() for line in \
    open(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/shape_names.txt'))] 

HOSTNAME = socket.gethostname()

# ModelNet40 official train/test split
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate(num_votes, sess1, ops1):
    is_training = False
     
    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        loss = MODEL.get_loss(pred, labels_pl, end_points)
        
        # Add ops to save and restore all the variables.
        saver = tf.train.import_meta_graph('log/model.ckpt.meta')
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'loss': loss,
           'x_in': ops1['x_in'],
           'net': ops1['net']}

    eval_one_epoch([sess, sess1], ops, num_votes)

   
def eval_one_epoch(sess, ops, num_votes=1, topk=1):
    error_cnt = 0
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    for fn in range(len(TEST_FILES)):
        log_string('----'+str(fn)+'----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        temp_data = np.zeros((len(current_data), NUM_POINT, 3))
        if SAMPLING == 'none':
            temp_data = current_data[:,0:NUM_POINT,:]
        elif SAMPLING == 'fps':
            for item in range(len(current_data)):
                temp_data[item] = farthest_point_sample(current_data[item], NUM_POINT, -1)
        elif SAMPLING == 'mine':
            for item in range(len(current_data)):
                temp_data[item] = my_point_sample(current_data[item], NUM_POINT, -1)
        elif SAMPLING == 'mine_neighbor':
            for item in range(len(current_data)):
                dist = distance.squareform(distance.pdist(current_data[item]))
                temp_data[item] = my_point_sample_neighbor(current_data[item], NUM_POINT, item, 128, dist)
        elif SAMPLING == 'featured':
            item_num = len(current_data)
            batch_num = item_num
            temp_data = np.empty([0, NUM_POINT, 3], float)
            for i in range(batch_num):
                start_idx = i * (item_num // batch_num)
                end_idx = (i + 1) * (item_num // batch_num)
                temp = my_point_sample_featured(sess[1], ops, "eval" + "_" + str(fn) + "_" + str(i), current_data[start_idx : end_idx], NUM_POINT, 8)
                # temp = np.reshape(temp, [NUM_POINT, 3])
                temp_data = np.append(temp_data, temp, axis=0)
        current_label = np.squeeze(current_label)
        
        file_size = temp_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        print(file_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            cur_batch_size = end_idx - start_idx
            
            # Aggregating BEG
            batch_loss_sum = 0 # sum of losses for the batch
            batch_pred_sum = np.zeros((cur_batch_size, NUM_CLASSES)) # score for classes
            batch_pred_classes = np.zeros((cur_batch_size, NUM_CLASSES)) # 0/1 for classes
            for vote_idx in range(num_votes):
                rotated_data = provider.rotate_point_cloud_by_angle(temp_data[start_idx:end_idx, :, :],
                                                  vote_idx/float(num_votes) * np.pi * 2)
                feed_dict = {ops['pointclouds_pl']: rotated_data,
                             ops['labels_pl']: current_label[start_idx:end_idx],
                             ops['is_training_pl']: is_training}
                loss_val, pred_val = sess[0].run([ops['loss'], ops['pred']],
                                          feed_dict=feed_dict)
                batch_pred_sum += pred_val
                batch_pred_val = np.argmax(pred_val, 1)
                for el_idx in range(cur_batch_size):
                    batch_pred_classes[el_idx, batch_pred_val[el_idx]] += 1
                batch_loss_sum += (loss_val * cur_batch_size / float(num_votes))
            # pred_val_topk = np.argsort(batch_pred_sum, axis=-1)[:,-1*np.array(range(topk))-1]
            # pred_val = np.argmax(batch_pred_classes, 1)
            pred_val = np.argmax(batch_pred_sum, 1)
            # Aggregating END
            
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            # correct = np.sum(pred_val_topk[:,0:topk] == label_val)
            total_correct += correct
            total_seen += cur_batch_size
            loss_sum += batch_loss_sum

            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
                fout.write('%d, %d\n' % (pred_val[i-start_idx], l))
                
                if pred_val[i-start_idx] != l and FLAGS.visu: # ERROR CASE, DUMP!
                    img_filename = '%d_label_%s_pred_%s.jpg' % (error_cnt, SHAPE_NAMES[l],
                                                           SHAPE_NAMES[pred_val[i-start_idx]])
                    img_filename = os.path.join(DUMP_DIR, img_filename)
                    output_img = pc_util.point_cloud_three_views(np.squeeze(temp_data[i, :, :]))
                    output_img = output_img.astype(np.uint8)
                    imageio.imwrite(img_filename, output_img)
                    # scipy.misc.imsave(img_filename, output_img)
                    error_cnt += 1
                
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f' % (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float64))))
    
    class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float64)
    for i, name in enumerate(SHAPE_NAMES):
        log_string('%10s:\t%0.3f' % (name, class_accuracies[i]))
    


if __name__=='__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    with tf.Graph().as_default() as g:
        item_num = 1
        item_point_num = 2048
        neighbor_num = 8
        x_in = tf.placeholder("float", [item_num, item_point_num, 2 * neighbor_num + 1, 3])
        # x = tf.reshape(x_in, [item_num, 2 * neighbor_num + 1, 3, 1])

        # print(x_in.shape)

        conv1_w = tf.get_variable("conv1_w", [1, 2 * neighbor_num + 1, 3, 64], initializer=tf.compat.v1.keras.initializers.glorot_normal())
        net = tf.nn.conv2d(x_in, conv1_w, [1, 1, 1, 1], "VALID")
        net = tf_util.batch_norm_for_conv2d(net, tf.constant(True), bn_decay=True, scope='eval_bn1')
        net = tf.nn.relu(net)
        # print(net.shape)

        conv2_w = tf.get_variable("conv2_w", [1, 1, 64, 64], initializer=tf.compat.v1.keras.initializers.glorot_normal())
        net = tf.nn.conv2d(net, conv2_w, [1, 1, 1, 1], "VALID")
        net = tf_util.batch_norm_for_conv2d(net, tf.constant(True), bn_decay=True, scope='eval_bn2')
        net = tf.nn.relu(net)
        # # print(net.shape)

        with tf.variable_scope('transform_net2') as sc:
            transform = feature_transform_net(net, tf.constant(True), bn_decay=True, K=64)
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        net_transformed = tf.expand_dims(net_transformed, [2])

        conv3_w = tf.get_variable("conv3_w", [1, 1, 64, 64], initializer=tf.compat.v1.keras.initializers.glorot_normal())
        net = tf.nn.conv2d(net_transformed, conv3_w, [1, 1, 1, 1], "VALID")
        net = tf_util.batch_norm_for_conv2d(net, tf.constant(True), bn_decay=True, scope='eval_bn3')
        # # print(net.shape)

        conv4_w = tf.get_variable("conv4_w", [1, 1, 64, 128], initializer=tf.compat.v1.keras.initializers.glorot_normal())
        net = tf.nn.conv2d(net, conv4_w, [1, 1, 1, 1], "VALID")
        net = tf_util.batch_norm_for_conv2d(net, tf.constant(True), bn_decay=True, scope='eval_bn4')
        net = tf.nn.relu(net)
        # print(net.shape)

        conv5_w = tf.get_variable("conv5_w", [1, 1, 128, 1024], initializer=tf.compat.v1.keras.initializers.glorot_normal())
        net = tf.nn.conv2d(net, conv5_w, [1, 1, 1, 1], "VALID")
        net = tf_util.batch_norm_for_conv2d(net, tf.constant(True), bn_decay=True, scope='eval_bn5')
        net = tf.nn.relu(net)
        # print(net.shape)

        max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(2 * neighbor_num + 1, 1), strides=(1, 1), padding="VALID", data_format="channels_last")
        max_pool_2d(net)
        # net = tf.nn.max_pool(net, [item_num, 2 * k + 1, 1, 1], [1, 1, 1, 1], padding='VALID')
        # print(net.shape)

        net = tf.reshape(net, [item_num, item_point_num, -1])
        # print(net.shape)

        fc1_w = tf.get_variable("fc1_w", [1024, 512], initializer=tf.compat.v1.keras.initializers.glorot_normal())
        net = tf.matmul(net, fc1_w)
        net = tf_util.batch_norm_for_fc(net, tf.constant(True), bn_decay=True, scope='eval_bn6')
        net = tf.nn.relu(net)
        # print(net.shape)

        net = tf.nn.dropout(net, 0.7)
        fc2_w = tf.get_variable("fc2_w", [512, 256], initializer=tf.compat.v1.keras.initializers.glorot_normal())
        net = tf.matmul(net, fc2_w)
        net = tf_util.batch_norm_for_fc(net, tf.constant(True), bn_decay=True, scope='eval_bn7')
        net = tf.nn.relu(net)
        # print(net.shape)

        net = tf.nn.dropout(net, 0.7)
        fc3_w = tf.get_variable("fc3_w", [256, 1], initializer=tf.compat.v1.keras.initializers.glorot_normal())
        net = tf.matmul(net, fc3_w)
        net = tf_util.batch_norm_for_fc(net, tf.constant(True), bn_decay=True, scope='eval_bn8')
        net = tf.nn.relu(net)
        ops = {
            'x_in': x_in,
            'net': net
        }

        sess1 = tf.Session(config=config)
        init = tf.global_variables_initializer()
        sess1.run(init)
        with sess1.as_default():
            saver = tf.train.import_meta_graph('log/sample.ckpt.meta')
            # saver.restore(sess1, tf.train.latest_checkpoint('/tmp/model-subset/'))
            saver.restore(sess1, SAMPLE_PATH)

    with tf.Graph().as_default():
        evaluate(num_votes=1, sess1=sess1, ops1=ops)
    LOG_FOUT.close()
