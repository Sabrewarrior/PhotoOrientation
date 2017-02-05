########################################################################################
# Edited by: Ujash Joshi, 2016                                                         #
# Based on: Davi Frossard, 2016                                                        #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# https://github.com/Sabrewarrior/photoorientationblob/master/vgg16.py                 #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                          simon frazer, guelph                                        #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import numpy as np
from scipy.misc import imread, imresize
import tensorflow as tf


class VGG16:
    def __init__(self, batch_size, learning_rate, max_pool_num=5, fc_size=4096, data_mean=None,guided_grad=False,
                 pre_fc=False, global_step=None, snapshot=None):
        if data_mean is None:
            data_mean = [123.68, 116.779, 103.939]
        self.data_mean = data_mean
        self.parameters = {}
        self.tensors = {}
        self.gradients = {}
        self.fc_size = fc_size
        self.global_step = global_step
        self.learning_rate = learning_rate
        self.batchsize = batch_size
        self.inputs = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3), name="Inputs")
        self.labels = tf.placeholder(tf.int32, shape=(batch_size), name="Outputs")
        self.testy = tf.placeholder(tf.int32, [batch_size, ], name="Test_y")
        self.keep_probs = tf.Variable(1, name='keep_probs', trainable=False, dtype=tf.float32)
        last_pool_name = self.create_conv_layers(snapshot, max_pool_num, pre_layer=pre_fc)
        if max_pool_num > 0:
            self.outputs = self.fc_layers(last_pool_name, snapshot)
            self.probs = tf.nn.softmax(self.outputs)
            self.prediction = tf.argmax(self.probs, 1)
            self.correct_predictions = tf.equal(self.prediction, tf.to_int64(self.testy))
            with tf.name_scope("Accuracy"):
                self.acc = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))

        if pre_fc:
            self.outputs1 = tf.nn.softmax(self.tensors["fc1"])
            self.probs1 = tf.nn.softmax(self.outputs1)
            self.prediction1 = tf.argmax(self.probs1, 1)
            self.correct_predictions1 = tf.equal(self.prediction1, tf.to_int64(self.testy))
            self.train_step1 = self.training(self.outputs1)
            with tf.name_scope("Accuracy_fc0"):
                self.acc1 = tf.reduce_mean(tf.cast(self.correct_predictions1, tf.float32))

        if guided_grad:
            self.gradients.update({"probs" : tf.gradients(self.probs, self.inputs)})
            self.gradients.update({"outputs": tf.gradients(self.outputs, self.inputs)})
            self.gradients.update({"preds": tf.gradients(tf.reduce_max(self.probs), self.inputs)})
            i = 0
            for each in tf.split(1, 4, self.probs):
                self.gradients.update({"prob"+str(i): tf.gradients(each, self.inputs)})
                i += 1
        elif max_pool_num > 0:
            # Do not train with changed relu
            self.train_step = self.training(self.outputs)

    def training(self, outputs):
        with tf.name_scope("Training"):
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(outputs, tf.to_int64(self.labels))
            cost = tf.reduce_mean(entropy)
            return tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=self.global_step)

    def convolve(self, layer_num, conv_shape_list, conv_stride, pool_ksize, input_name, snapshot):
        num_convs = len(conv_shape_list)
        with tf.name_scope("conv" + str(layer_num)) as scope:
            for conv_num in range(1, num_convs+1):
                cur_conv = "conv" + str(layer_num) + "_" + str(conv_num)
                print(cur_conv)
                if snapshot:
                    print("Snapshot found for " + cur_conv + " loading weights and biases")
                    kernel = snapshot[cur_conv + "_W"]
                    biases = snapshot[cur_conv + "_b"]
                else:
                    kernel = tf.truncated_normal(conv_shape_list[conv_num - 1], dtype=tf.float32, stddev=1e-1)
                    biases = tf.constant(1.0, shape=[conv_shape_list[conv_num - 1][-1]], dtype=tf.float32)

                self.parameters.update({cur_conv + "_W": tf.Variable(kernel, name="weights")})
                self.parameters.update({cur_conv + "_b": tf.Variable(biases, trainable=True, name="weights")})

                conv = tf.nn.conv2d(self.tensors[input_name], self.parameters[cur_conv+"_W"],
                                    conv_stride, padding='SAME')
                out = tf.nn.bias_add(conv, self.parameters[cur_conv+"_b"])
                self.tensors.update({cur_conv: tf.nn.relu(out, name="activation_" + str(conv_num))})
                input_name = cur_conv
                self.gradients.update({input_name: tf.gradients(self.tensors[input_name], self.inputs)})

            return tf.nn.max_pool(self.tensors[input_name], ksize=pool_ksize, strides=pool_ksize, padding='SAME',
                                  name='pool')

    def create_conv_layers(self, snapshot, pool_num=5, pre_layer=False):
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant(self.data_mean, dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            input_name = "pre_proc_images"
            self.tensors.update({input_name: tf.sub(self.inputs, mean)})

        if pre_layer:
            with tf.name_scope('fc0') as scope:
                #shape = int(np.prod(self.tensors[input_name].get_shape()[1:]))
                #print(shape)
                #flat_image = tf.reshape(self.tensors[input_name], [-1, shape])
                if snapshot and "fc0_W" in snapshot and (150528, 150528) == snapshot['fc0_W'].shape:
                    wl = snapshot['fc0_W']
                    bl = snapshot['fc0_b']
                    print("Snapshot found for fc0, loading weights and biases")
                else:
                    wl = tf.truncated_normal([self.batchsize, 224, 3, 3], dtype=tf.float32, stddev=1e-1)
                    print(wl.get_shape())
                    # wl = tf.reshape()
                    bl = tf.ones(shape=[3], dtype=tf.float32)
                self.parameters.update({"fc0_W": tf.Variable(wl, trainable=True, name='weights')})
                self.parameters.update({"fc0_b": tf.Variable(bl, trainable=True, name='biases')})

                fc0l = tf.nn.bias_add(tf.matmul(self.tensors[input_name] , self.parameters['fc0_W']),
                                      self.parameters['fc0_b'])
                self.tensors.update({'fc0': tf.nn.dropout(tf.nn.relu(fc0l, name="activation"), self.keep_probs)})
                self.gradients.update({"fc0": tf.gradients(self.tensors["fc0"], self.inputs)})
                input_name = "fc0"

            # fc1
            with tf.name_scope('fc1') as scope:
                fc0_shape = int(np.prod(self.tensors[input_name].get_shape()[1:]))
                print(fc0_shape)
                flat_fc0 = tf.reshape(self.tensors[input_name], [-1, fc0_shape])
                if snapshot and "fc1_W" in snapshot and 150528 == snapshot['fc1_W'].shape[0] and \
                                4 == snapshot['fc1_W'].shape[1]:
                    print("Snapshot found for fc1, loading weights and biases")
                    wl = snapshot['fc1_W']
                    bl = snapshot['fc1_b']
                else:
                    wl = tf.truncated_normal([fc0_shape, 4], dtype=tf.float32, stddev=1e-1)
                    bl = tf.zeros(shape=[4], dtype=tf.float32)
                self.parameters.update({"fc1_W": tf.Variable(wl, trainable=True, name='weights')})
                self.parameters.update({"fc1_b": tf.Variable(bl, trainable=True, name='biases')})
                self.tensors.update({"fc1": tf.nn.bias_add(tf.matmul(flat_fc0, self.parameters['fc1_W']),
                                                           self.parameters['fc1_b'])})


        if pool_num > 0:
            with tf.name_scope('conv1') as scope:
                self.tensors.update({"pool1": self.convolve(1, [[3, 3, 3, 64], [3, 3, 64, 64]],
                                                            [1, 1, 1, 1], [1, 2, 2, 1], input_name, snapshot)})
                input_name = "pool1"
            self.gradients.update({input_name: tf.gradients(self.tensors[input_name], self.inputs)})

        if pool_num > 1:
            with tf.name_scope('conv2') as scope:
                self.tensors.update({"pool2": self.convolve(2, [[3, 3, 64, 128],[3, 3, 128, 128]],
                                                            [1, 1, 1, 1], [1, 2, 2, 1], input_name, snapshot)})
                input_name = "pool2"
        else:
            with tf.name_scope('conv2') as scope:
                self.tensors.update({"pool2": tf.nn.max_pool(self.tensors[input_name], ksize=[1, 1, 1, 1],
                                                             strides=[1, 2, 2, 1], padding='SAME', name='pool')})
                input_name = "pool2"
        self.gradients.update({input_name: tf.gradients(self.tensors[input_name], self.inputs)})

        if pool_num > 2:
            with tf.name_scope('conv3') as scope:
                self.tensors.update({"pool3": self.convolve(3, [[3, 3, 128, 256],[3, 3, 256, 256],[3, 3, 256, 256]],
                                                            [1, 1, 1, 1], [1, 2, 2, 1], input_name, snapshot)})
                input_name = "pool3"
        else:
            with tf.name_scope('conv3') as scope:
                self.tensors.update({"pool3": tf.nn.max_pool(self.tensors[input_name], ksize=[1, 1, 1, 1],
                                                             strides=[1, 2, 2, 1], padding='SAME', name='pool')})
                input_name = "pool3"
        self.gradients.update({input_name: tf.gradients(self.tensors[input_name], self.inputs)})

        if pool_num > 3:
            with tf.name_scope('conv4') as scope:
                self.tensors.update({"pool4": self.convolve(4, [[3, 3, 256, 512],[3, 3, 512, 512],[3, 3, 512, 512]],
                                                            [1, 1, 1, 1], [1, 2, 2, 1], input_name, snapshot)})
                input_name = "pool4"
        else:
            with tf.name_scope('conv4') as scope:
                self.tensors.update({"pool4": tf.nn.max_pool(self.tensors[input_name], ksize=[1, 1, 1, 1],
                                                             strides=[1, 2, 2, 1], padding='SAME', name='pool')})
                input_name = "pool4"
        self.gradients.update({input_name: tf.gradients(self.tensors[input_name], self.inputs)})

        if pool_num > 4:
            with tf.name_scope('conv5') as scope:
                self.tensors.update({"pool5": self.convolve(5, [[3, 3, 512, 512],[3, 3, 512, 512],[3, 3, 512, 512]],
                                                            [1, 1, 1, 1], [1, 2, 2, 1], input_name, snapshot)})
                input_name = "pool5"
        else:
            with tf.name_scope('conv5') as scope:
                self.tensors.update({"pool5": tf.nn.max_pool(self.tensors[input_name], ksize=[1, 1, 1, 1],
                                                             strides=[1, 2, 2, 1], padding='SAME', name='pool')})
                input_name = "pool5"
        self.gradients.update({input_name: tf.gradients(self.tensors[input_name], self.inputs)})

        if pool_num > 5:
            with tf.name_scope('conv6') as scope:
                self.tensors.update({"pool6": self.convolve(5,[[3, 3, 512, 1024],[3, 3, 1024, 1024],[3, 3, 1024, 1024]],
                                                            [1, 1, 1, 1], [1, 2, 2, 1], input_name, snapshot)})
                input_name = "pool6"
                self.gradients.update({input_name: tf.gradients(self.tensors[input_name], self.inputs)})
        return input_name

    def fc_layers(self, input_name, snapshot):
        shape = int(np.prod(self.tensors[input_name].get_shape()[1:]))
        final_pool_flat = tf.reshape(self.tensors[input_name], [-1, shape])
        print("Shape of last conv is " + str(shape))
        with tf.name_scope('fc6') as scope:
            if snapshot and (shape, self.fc_size) == snapshot['fc6_W'].shape:
                print(snapshot['fc6_W'].shape, shape)
                print("Snapshot found for fc6, loading weights and biases")
                wl = snapshot['fc6_W']
                bl = snapshot['fc6_b']
            else:
                wl = tf.truncated_normal([shape, self.fc_size], dtype=tf.float32, stddev=1e-1)
                bl = tf.constant(1.0, shape=[self.fc_size], dtype=tf.float32)
            self.parameters.update({"fc6_W": tf.Variable(wl, trainable=True, name='weights')})
            self.parameters.update({"fc6_b": tf.Variable(bl, trainable=True, name='biases')})
            fc6l = tf.nn.bias_add(tf.matmul(final_pool_flat, self.parameters['fc6_W']), self.parameters['fc6_b'])
            self.tensors.update({'fc6': tf.nn.dropout(tf.nn.relu(fc6l, name="activation"), self.keep_probs)})
            self.gradients.update({"fc6": tf.gradients(self.tensors["fc6"], self.inputs)})
        # fc7
        with tf.name_scope('fc7') as scope:
            if snapshot and (self.fc_size, self.fc_size) == snapshot['fc7_W'].shape:
                wl = snapshot['fc7_W']
                bl = snapshot['fc7_b']
                print("Snapshot found for fc7, loading weights and biases")
            else:
                wl = tf.truncated_normal([self.fc_size, self.fc_size], dtype=tf.float32, stddev=1e-1)
                bl = tf.constant(1.0, shape=[self.fc_size], dtype=tf.float32)
            self.parameters.update({"fc7_W": tf.Variable(wl, trainable=True, name='weights')})
            self.parameters.update({"fc7_b": tf.Variable(bl, trainable=True, name='biases')})

            fc7l = tf.nn.bias_add(tf.matmul(self.tensors['fc6'], self.parameters['fc7_W']), self.parameters['fc7_b'])
            self.tensors.update({'fc7': tf.nn.dropout(tf.nn.relu(fc7l, name="activation"), self.keep_probs)})
            self.gradients.update({"fc7": tf.gradients(self.tensors["fc7"], self.inputs)})

        # fc8
        with tf.name_scope('fc8') as scope:
            if snapshot and self.fc_size == snapshot['fc8_W'].shape[0] and 4 == snapshot['fc8_W'].shape[1]:
                print("Snapshot found for fc8, loading weights and biases")
                wl = snapshot['fc8_W']
                bl = snapshot['fc8_b']
            else:
                wl = tf.truncated_normal([self.fc_size, 4], dtype=tf.float32, stddev=1e-1)
                bl = tf.constant(0.1, shape=[4], dtype=tf.float32)
            self.parameters.update({"fc8_W": tf.Variable(wl, trainable=True, name='weights')})
            self.parameters.update({"fc8_b": tf.Variable(bl, trainable=True, name='biases')})

            return tf.nn.bias_add(tf.matmul(self.tensors['fc7'], self.parameters['fc8_W']), self.parameters['fc8_b'])


if __name__ == '__main__':
    test = False
    # This will not work with edited VGG
    if test:
        sess = tf.Session()
        batchSize = 1000
        globalStep = tf.Variable(0, name='global_step', trainable=False)
        imgs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name="Inputs")
        y_ = tf.placeholder(tf.int32,shape=(batchSize), name="Outputs")
        learning_rate = .0001
        M = np.load('vgg16_weights.npz')
        vgg = VGG16(imgs, y_, learning_rate, max_pool_num=5, global_step=globalStep, snapshot=M)
        init = tf.initialize_all_variables()
        sess.run(init)

        img1 = imread('laska.png', mode='RGB')
        img1 = imresize(img1, (224, 224))

        prob = sess.run(vgg.probs, feed_dict={vgg.inputs: [img1]})[0]
        print(prob)

