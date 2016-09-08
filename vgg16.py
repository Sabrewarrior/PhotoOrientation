########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names
import os
import tensorflow as tf


class VGG16:
    def __init__(self, imgs, y_, learning_rate, global_step=None, snapshot=None):
        self.inputs = imgs
        self.labels = y_
        self.parameters = {}
        self.tensors = {}
        self.global_step = global_step
        self.create_conv_layers(snapshot)

        self.outputs = self.fc_layers("pool5",snapshot)
        self.learning_rate = learning_rate
        self.train_step = self.training()
        self.acc = self.accuracy()

    def accuracy(self):
        with tf.name_scope("Accuracy"):
            self.testy = tf.placeholder(tf.int32, [None, ], name="Test_y")
            self.probs = tf.nn.softmax(self.outputs)
            correct_prediction = tf.equal(tf.argmax(self.probs, 1), tf.to_int64(self.testy))
            return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
                    kernel = snapshot[cur_conv + "_W"]
                    biases = snapshot[cur_conv + "_b"]
                else:
                    kernel = tf.truncated_normal(conv_shape_list[conv_num - 1], dtype=tf.float32, stddev=1e-1)
                    biases = tf.constant(0.0, shape=[conv_shape_list[conv_num -1][-1]], dtype=tf.float32)
                print(kernel.shape)
                print(biases.shape)
                self.parameters.update({cur_conv + "_W": tf.Variable(kernel, name="weights")})
                self.parameters.update({cur_conv + "_b": tf.Variable(biases, trainable=True, name="weights")})

                conv = tf.nn.conv2d(self.tensors[input_name], self.parameters[cur_conv+"_W"],
                                    conv_stride, padding='SAME')
                out = tf.nn.bias_add(conv, self.parameters[cur_conv+"_b"])
                self.tensors.update({cur_conv: tf.nn.relu(out, name="activation_"+ str(conv_num))})
                input_name = cur_conv

            return tf.nn.max_pool(self.tensors[input_name], ksize=pool_ksize, strides=pool_ksize, padding='SAME', name='pool')

    def create_conv_layers(self, snapshot):
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            input_name = "pre_proc_images"
            self.tensors.update({input_name: tf.sub(self.inputs, mean)})

        with tf.name_scope('conv1') as scope:
            self.tensors.update({"pool1": self.convolve(1, [[3,3,3,64],[3,3,64,64]],
                                                  [1,1,1,1], [1,2,2,1], input_name, snapshot)})
            input_name = "pool1"

        with tf.name_scope('conv2') as scope:
            self.tensors.update({"pool2": self.convolve(2, [[3, 3, 64, 128],[3, 3, 128, 128]],
                                                  [1,1,1,1], [1,2,2,1], input_name, snapshot)})
            input_name = "pool2"

        with tf.name_scope('conv3') as scope:
            self.tensors.update({"pool3": self.convolve(3, [[3, 3, 128, 256],[3, 3, 256, 256],[3, 3, 256, 256]],
                                                  [1,1,1,1], [1,2,2,1], input_name, snapshot)})
            input_name = "pool3"

        with tf.name_scope('conv4') as scope:
            self.tensors.update({"pool4": self.convolve(4, [[3, 3, 256, 512],[3, 3, 512, 512],[3, 3, 512, 512]],
                                                  [1,1,1,1], [1,2,2,1], input_name, snapshot)})
            input_name = "pool4"

        with tf.name_scope('conv5') as scope:
            self.tensors.update({"pool5": self.convolve(5, [[3, 3, 512, 512],[3, 3, 512, 512],[3, 3, 512, 512]],
                                                  [1,1,1,1], [1,2,2,1], input_name, snapshot)})
            input_name = "pool5"

        '''
        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            cur_conv = "conv" + str(layer_num) + "_" + str(conv_num) + "_"
            if snapshot:
                kernel = snapshot[cur_conv + "W"]
                biases = snapshot[cur_conv + "b"]
            else:
                kernel = tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1)
                biases = tf.constant(0.0, shape=[64], dtype=tf.float32)
            self.parameters.update({cur_conv + "W": tf.Variable(kernel,trainable=True, name="weights")})
            self.parameters.update({cur_conv + "b": tf.Variable(biases, trainable=True, name="weights")})
            conv = tf.nn.conv2d(images, self.parameters[cur_conv+"W"], [1, 1, 1, 1], padding='SAME')

            out = tf.nn.bias_add(conv, self.parameters[cur_conv+"b"])
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.tensors["conv1_1"] = tf.nn.relu(out, name=scope)


        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')
        '''

        return input_name

    def fc_layers(self, input_name, snapshot):
        shape = int(np.prod(self.tensors[input_name].get_shape()[1:]))
        pool5_flat = tf.reshape(self.tensors[input_name], [-1, shape])

        with tf.name_scope('fc6') as scope:
            if snapshot:
                wl = snapshot['fc6_W']
                bl = snapshot['fc6_b']
            else:
                wl = tf.truncated_normal([shape, 4096], dtype=tf.float32, stddev=1e-1)
                bl = tf.constant(1.0, shape=[4096], dtype=tf.float32)
            self.parameters.update({"fc6_W": tf.Variable(wl, name='weights')})
            self.parameters.update({"fc6_b": tf.Variable(bl, trainable=True, name='biases')})

            fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, self.parameters['fc6_W']), self.parameters['fc6_b'])
            self.tensors.update({'fc6': tf.nn.relu(fc6l, name="activation")})


        # fc7
        with tf.name_scope('fc7') as scope:
            if snapshot:
                wl = snapshot['fc7_W']
                bl = snapshot['fc7_b']
            else:
                wl = tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1)
                bl = tf.constant(1.0, shape=[4096], dtype=tf.float32)
            self.parameters.update({"fc7_W": tf.Variable(wl, name='weights')})
            self.parameters.update({"fc7_b": tf.Variable(bl, trainable=True, name='biases')})

            fc7l = tf.nn.bias_add(tf.matmul(self.tensors['fc6'], self.parameters['fc7_W']), self.parameters['fc7_b'])
            self.tensors.update({'fc7': tf.nn.relu(fc7l, name="activation")})



        # fc8
        with tf.name_scope('fc8') as scope:
            if snapshot:
                wl = snapshot['fc8_W']
                bl = snapshot['fc8_b']
            else:
                wl = tf.truncated_normal([4096, 1000], dtype=tf.float32)
                bl = tf.constant(1.0, shape=[1000], dtype=tf.float32)
            self.parameters.update({"fc8_W": tf.Variable(wl, name='weights')})
            self.parameters.update({"fc8_b": tf.Variable(bl, trainable=True, name='biases')})

            return tf.nn.bias_add(tf.matmul(self.tensors['fc7'], self.parameters['fc8_W']), self.parameters['fc8_b'])


if __name__ == '__main__':
    test = False
    if test:
        sess = tf.Session()
        batchSize = 1000
        globalStep = tf.Variable(0, name='global_step', trainable=False)
        imgs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name="Inputs")
        y_ = tf.placeholder(tf.int32,shape=(batchSize), name="Outputs")
        learning_rate = .0001
        M = np.load('vgg16_weights.npz')
        vgg = VGG16(imgs, y_, learning_rate, global_step=globalStep, snapshot=M)
        init = tf.initialize_all_variables()
        sess.run(init)

        img1 = imread('laska.png', mode='RGB')
        img1 = imresize(img1, (224, 224))

        prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
        preds = (np.argsort(prob)[::-1])[0:5]
        for p in preds:
            print(class_names[p], prob[p])


