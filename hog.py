import numpy as np
import time
import random
from datahandler import create_labeled_image_list, convert_binary_to_array
import os
import pickle
import tensorflow as tf
from skimage.feature import hog
from skimage.color import rgb2gray
from tensorflow.python.framework import dtypes, ops

t = int(time.time())
# t = 1454219613
print("t=", t)
random.seed(t)


'''
def read_my_file_format(filename_queue):
    reader = tf.WholeFileReader()
    key, record_string = reader.read(filename_queue[0])
    image = tf.image.decode_jpeg(record_string)
    label = filename_queue[1]
    return image, label
'''


def read_hog_file_format(filename_queue):
    image = tf.read_file(filename_queue[0])
    label = tf.cast(filename_queue[1], tf.int32)
    return image, label


def input_pipeline(directory, batch_size, data_set="train", feature="images", num_epochs=None, num_images=None):
    with tf.name_scope('input'):
        # Reads paths of images together with their labels
        image_list, label_list = create_labeled_image_list(directory, data_set=data_set, feature=feature,
                                                           num_images=num_images)

        # Makes an input queue
        input_queue = tf.train.slice_input_producer([image_list, label_list],num_epochs=num_epochs)
        image, label = read_hog_file_format(input_queue)

        # min_after_dequeue defines how big a buffer we will randomly sample
        #   from -- bigger means better shuffling but slower start up and more
        #   memory used.
        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * batch_size
        if data_set.count("train") == 1:
            image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue, num_threads=1)
        else:
            image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, capacity=capacity,
                                  num_threads=1)
        return image_batch, label_batch


# TODO: Normalize HOG?
def hog1layer(batch_size, learning_rate, data_folder, snapshot = None, feature="hog"):
    feature_num = 1764
    hid = 2560
    with tf.Graph().as_default():
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        globalStep = tf.Variable(0, name='global_step', trainable=False)

        # Use for test sets
        testy = tf.placeholder(tf.int32, [None, ], name="Test_y")
        # xtest = tf.placeholder(tf.float32, [None, 1764])
        x = tf.placeholder(tf.float32, shape=(None,feature_num), name="Input")
        y_ = tf.placeholder(tf.int32,shape=(batch_size), name="Output")

        image_batch, label_batch = input_pipeline(data_folder, batch_size, data_set="train", feature=feature)
        test_images, test_labels = input_pipeline(data_folder, 1000, data_set="test", feature=feature, num_images=12000)
        valid_images, valid_labels = input_pipeline(data_folder, 1000, data_set="valid", feature=feature,
                                                    num_images=12000)
        #with tf.device('/cpu:0'):
        if not snapshot:
            print("1")
            w0 = tf.Variable(tf.random_normal([feature_num, hid], dtype=tf.float32, stddev=1e-1))
            b0 = tf.Variable(tf.random_normal([hid], dtype=tf.float32, stddev=1e-1))
        else:
            w0 = tf.Variable(snapshot["w0"])
            b0 = tf.Variable(snapshot["b0"])
        layer1l = tf.add(tf.matmul(x, w0), b0)
        layer1 = tf.nn.tanh(layer1l)
        if not snapshot:
            print("2")
            w1 = tf.Variable(tf.random_normal([hid, 4], dtype=tf.float32, stddev=1e-1))
            b1 = tf.Variable(tf.random_normal([4], dtype=tf.float32, stddev=1e-1))
        else:
            w1 = tf.Variable(snapshot["w1"])
            b1 = tf.Variable(snapshot["b1"])
        prediction = tf.add(tf.matmul(layer1, w1), b1)



        #lam = 0.00000
        #decay_penalty = tf.add(tf.mul(lam, tf.reduce_sum(tf.square(w0))), tf.mul(lam, tf.reduce_sum(tf.square(w1))))
        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(prediction, tf.to_int64(y_))
        #cost = tf.add(tf.reduce_mean(entropy), decay_penalty)
        cost = tf.reduce_mean(entropy)

        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost, global_step=globalStep)

        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.to_int64(testy))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init = tf.initialize_all_variables()
        sess.run(init)

        # test_x, test_y = input_pipeline(os.path.join(data_folder, "test"), batchSize, num_epochs=1)

        timers = {"batching": 0., "converting": 0., "training": 0., "testing":0., "acc":0., "total_tests": 0.}

        snapshot = {}
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        steps = 0
        try:
            while not coord.should_stop():

                now = time.time()
                imgs, labels = sess.run([image_batch, label_batch])
                timers["batching"] += (time.time() - now)

                now = time.time()
                imgs = convert_binary_to_array(imgs)
                timers["converting"] += (time.time() - now)

                now = time.time()
                sess.run(train_step, feed_dict={x: imgs, y_: labels})
                timers["training"] += (time.time() - now)
                steps += 1
                #print(globalStep)
                if steps % 1000 == 0:
                    print(steps)

                    print("Train: " + str(sess.run(accuracy, feed_dict={x: imgs, testy: labels})))
                    acc = 0.
                    total_test = 0
                    now = time.time()
                    for i in range(12):
                        imgs_test, labels_test = sess.run([test_images, test_labels])
                        imgs_test = convert_binary_to_array(imgs_test)
                        total_test += len(imgs_test)
                        acc += sess.run(accuracy, feed_dict={x: imgs_test, testy: labels_test})
                    timers["testing"] += (time.time() - now)
                    timers["total_tests"] += 1

                    print("Test: " + str(acc/12))

                    if steps%10000 == 0:
                        acc_valid = 0.
                        for i in range(12):
                            imgs_valid, labels_valid = sess.run([valid_images, valid_labels])
                            imgs_valid = convert_binary_to_array(imgs_valid)
                            acc_valid += sess.run(accuracy, feed_dict={x: imgs_valid, testy: labels_valid})
                        print("Valid: " + str(acc_valid/12))
                        snapshot["w0"] = sess.run(w0)
                        snapshot["w1"] = sess.run(w1)
                        snapshot["b0"] = sess.run(b0)
                        snapshot["b1"] = sess.run(b1)
                        pickle.dump(snapshot,
                                open(os.path.join(data_folder, "snapshotHOG" + str(steps // 10000) + ".pkl"), "wb"))
                    if (acc/12)>.99:
                        break
                    # snapshot = {}
                #timers.append(time.time() - now)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        coord.join(threads)

        snapshot["w0"] = sess.run(w0)
        snapshot["w1"] = sess.run(w1)
        snapshot["b0"] = sess.run(b0)
        snapshot["b1"] = sess.run(b1)
        pickle.dump(snapshot,
                    open(os.path.join(data_folder, "snapshotHOGFinal.pkl"), "wb"))

        sess.close()
        for timer in timers.keys():
            if timer.count("acc") == 0:
                print(timer + " avg: " + str(timers[timer]/steps))
        print("acc avg: " + str(timers["acc"]/timers["total_tests"]))


if __name__ == "__main__":

    datafolder = "/home/ujash/nvme/data2"
    M = pickle.load(open(os.path.join(datafolder,"snapshotHOG458.pkl"),'rb'))
    hog1layer(1000, .00001, datafolder, feature="hog2", snapshot=M)

'''

im1 = (imread("421.jpg")[:,:,:3]).astype(float32)
im1 = im1/255.

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)


ax1.axis('off')
ax1.imshow(im1, cmap=plt.cm.gray)
ax1.set_title('Input image')
ax1.set_adjustable('box-forced')

im1 = rgb2gray(im1).astype(float32)

fd, hog_image = hog(im1, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
print fd.shape

fd, hog_image = hog(im1, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(1, 1), visualise=True)
print fd.shape

fd, hog_image = hog(im1, orientations=9, pixels_per_cell=(16, 16),
                    cells_per_block=(3, 3), visualise=True)
print fd.shape

fd, hog_image = hog(im1, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(3, 3), visualise=True)
print fd.shape

fd = hog(im1, orientations=15, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)[0].astype(float32)
print fd.shape
print fd.dtype
# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
imsave('hog.hog',hog_image_rescaled,format='JPEG')

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box-forced')
plt.show()
'''
