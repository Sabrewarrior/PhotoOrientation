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
from imagenet_classes import class_names
import os
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

class VGG16:
    def __init__(self, imgs, y_, learning_rate, max_pool_num=5, guided_grad=False, global_step=None, snapshot=None):

        self.inputs = imgs
        self.labels = y_
        self.testy = tf.placeholder(tf.int32, [None, ], name="Test_y")
        self.parameters = {}
        self.tensors = {}
        self.global_step = global_step
        self.learning_rate = learning_rate
        self.keep_probs = tf.Variable(1, name='keep_probs', trainable=False, dtype=tf.float32)
        if guided_grad:
            @ops.RegisterGradient("GuidedRelu")
            def _guided_relu_grad(op, grad):
                return tf.select(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))

            with tf.Graph().as_default() as g:
                with g.gradient_override_map({'Relu': 'GuidedRelu'}):
                    last_pool_name = self.create_conv_layers(snapshot, max_pool_num)
                    self.outputs = self.fc_layers(last_pool_name, snapshot)
        else:
            last_pool_name = self.create_conv_layers(snapshot, max_pool_num)
            self.outputs = self.fc_layers(last_pool_name, snapshot)
        self.probs = tf.nn.softmax(self.outputs)
        self.correct_predictions = tf.equal(tf.argmax(self.probs, 1), tf.to_int64(self.testy))
        with tf.name_scope("Accuracy"):
            self.acc = tf.reduce_mean(tf.cast(self.correct_predictions, tf.float32))

        self.train_step = self.training()

    def training(self):
        with tf.name_scope("Training"):
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(self.outputs, tf.to_int64(self.labels))
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
                    biases = tf.constant(0.0, shape=[conv_shape_list[conv_num -1][-1]], dtype=tf.float32)

                self.parameters.update({cur_conv + "_W": tf.Variable(kernel, name="weights")})
                self.parameters.update({cur_conv + "_b": tf.Variable(biases, trainable=True, name="weights")})

                conv = tf.nn.conv2d(self.tensors[input_name], self.parameters[cur_conv+"_W"],
                                    conv_stride, padding='SAME')
                out = tf.nn.bias_add(conv, self.parameters[cur_conv+"_b"])
                self.tensors.update({cur_conv: tf.nn.relu(out, name="activation_" + str(conv_num))})
                input_name = cur_conv

            return tf.nn.max_pool(self.tensors[input_name], ksize=pool_ksize, strides=pool_ksize, padding='SAME', name='pool')

    def create_conv_layers(self, snapshot, pool_num=5):
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            input_name = "pre_proc_images"
            self.tensors.update({input_name: tf.sub(self.inputs, mean)})

        with tf.name_scope('conv1') as scope:
            self.tensors.update({"pool1": self.convolve(1, [[3, 3, 3, 64], [3, 3, 64, 64]],
                                                        [1, 1, 1, 1], [1, 2, 2, 1], input_name, snapshot)})
            input_name = "pool1"

        with tf.name_scope('conv2') as scope:
            self.tensors.update({"pool2": self.convolve(2, [[3, 3, 64, 128],[3, 3, 128, 128]],
                                                        [1, 1, 1, 1], [1, 2, 2, 1], input_name, snapshot)})
            input_name = "pool2"

        with tf.name_scope('conv3') as scope:
            self.tensors.update({"pool3": self.convolve(3, [[3, 3, 128, 256],[3, 3, 256, 256],[3, 3, 256, 256]],
                                                        [1, 1, 1, 1], [1, 2, 2, 1], input_name, snapshot)})
            input_name = "pool3"
        if pool_num > 3:
            with tf.name_scope('conv4') as scope:
                self.tensors.update({"pool4": self.convolve(4, [[3, 3, 256, 512],[3, 3, 512, 512],[3, 3, 512, 512]],
                                                            [1, 1, 1, 1], [1, 2, 2, 1], input_name, snapshot)})
                input_name = "pool4"

        if pool_num > 4:
            with tf.name_scope('conv5') as scope:
                self.tensors.update({"pool5": self.convolve(5, [[3, 3, 512, 512],[3, 3, 512, 512],[3, 3, 512, 512]],
                                                            [1, 1, 1, 1], [1, 2, 2, 1], input_name, snapshot)})
                input_name = "pool5"

        return input_name

    def fc_layers(self, input_name, snapshot):
        shape = int(np.prod(self.tensors[input_name].get_shape()[1:]))
        final_pool_flat = tf.reshape(self.tensors[input_name], [-1, shape])
        print("Shape of last conv is " + str(shape))
        with tf.name_scope('fc6') as scope:
            if snapshot and shape == snapshot['fc6_W'].shape[0]:
                print("Snapshot found for fc6, loading weights and biases")
                wl = snapshot['fc6_W']
                bl = snapshot['fc6_b']
            else:
                wl = tf.truncated_normal([shape, 512], dtype=tf.float32, stddev=1e-1)
                bl = tf.constant(1.0, shape=[512], dtype=tf.float32)
            self.parameters.update({"fc6_W": tf.Variable(wl, trainable=True, name='weights')})
            self.parameters.update({"fc6_b": tf.Variable(bl, trainable=True, name='biases')})
            fc6l = tf.nn.bias_add(tf.matmul(final_pool_flat, self.parameters['fc6_W']), self.parameters['fc6_b'])
            self.tensors.update({'fc6': tf.nn.dropout(tf.nn.relu(fc6l, name="activation"), self.keep_probs)})

        # fc7
        with tf.name_scope('fc7') as scope:
            if snapshot and shape == snapshot['fc6_W'].shape[0]:
                wl = snapshot['fc7_W']
                bl = snapshot['fc7_b']
                print("Snapshot found for fc7, loading weights and biases")
            else:
                wl = tf.truncated_normal([512, 512], dtype=tf.float32, stddev=1e-1)
                bl = tf.constant(1.0, shape=[512], dtype=tf.float32)
            self.parameters.update({"fc7_W": tf.Variable(wl, trainable=True, name='weights')})
            self.parameters.update({"fc7_b": tf.Variable(bl, trainable=True, name='biases')})

            fc7l = tf.nn.bias_add(tf.matmul(self.tensors['fc6'], self.parameters['fc7_W']), self.parameters['fc7_b'])
            self.tensors.update({'fc7': tf.nn.dropout(tf.nn.relu(fc7l, name="activation"), self.keep_probs)})

        # fc8
        with tf.name_scope('fc8') as scope:
            if snapshot and shape == snapshot['fc6_W'].shape[0]:
                print("Snapshot found for fc8, loading weights and biases")
                wl = snapshot['fc8_W']
                bl = snapshot['fc8_b']
            else:
                wl = tf.truncated_normal([512, 4], dtype=tf.float32, stddev=1e-1)
                bl = tf.constant(0.1, shape=[4], dtype=tf.float32)
            self.parameters.update({"fc8_W": tf.Variable(wl, trainable=True, name='weights')})
            self.parameters.update({"fc8_b": tf.Variable(bl, trainable=True, name='biases')})

            return tf.nn.bias_add(tf.matmul(self.tensors['fc7'], self.parameters['fc8_W']), self.parameters['fc8_b'])


if __name__ == '__main__':
    test = True
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

