import numpy as np
import time
import random
from datahandler import load_test, load_train
import os
import tensorflow as tf

t = int(time.time())
#t = 1454219613
print "t=", t
random.seed(t)

def read_my_file_format(filename_queue):
    reader = tf.WholeFileReader()
    key, record_string = reader.read(filename_queue[0])
    image = tf.image.decode_jpeg(record_string)
    label = filename_queue[1]
    return image, label


def read_hog_file_format(filename_queue):

    print(filename_queue)
    filename, label = tf.decode_csv(filename_queue,[[""], [""]], " ")
    image = tf.read_file(filename)


    return image, label

def create_labeled_image_list(directory):
    image_list = []
    #label_list = []
    for root, dirnames, filenames in os.walk(directory):
        for dirname in dirnames:
            if dirname == '0':
                label = 0
            elif dirname == '90':
                label = 1
            elif dirname == '180':
                label = 2
            else:
                label = 3
            for root, dirnames_inner, filenames_actual in os.walk(os.path.join(root,dirname)):
                for filename in filenames_actual:
                    image_list.append(filename+ " " + str(label))
                    #label_list.append(label)
        break
    return image_list

def input_pipeline(directory, batch_size, num_epochs=None):
    with tf.name_scope('input'):
        # Reads paths of images together with their labels
        image_list = create_labeled_image_list(directory)

        # Makes an input queue
        input_queue = tf.train.string_input_producer(image_list)
        image, label = read_hog_file_format(input_queue)

        # min_after_dequeue defines how big a buffer we will randomly sample
        #   from -- bigger means better shuffling but slower start up and more
        #   memory used.
        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * batch_size
        image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                  min_after_dequeue=min_after_dequeue, num_threads=10)
        return image_batch, label_batch


def layer(x, weights, biases):
    return tf.nn.tanh(tf.add(tf.matmul(x, weights), biases))

def initWeightBias(x_dims, y_dims, weights=None, biases=None):
    if not weights:
        weights = tf.Variable(tf.random_normal([x_dims, y_dims], stddev=0.01))
    if not biases:
        biases = tf.Variable(tf.random_normal([y_dims], stddev=0.01))
    return weights, biases

def hog1layer(batchSize, learningRate, data_folder):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    with tf.Graph().as_default():
        globalStep = tf.Variable(0, name='global_step', trainable=False)

        x = tf.placeholder(tf.float32, shape=(batchSize,1764))
        y_ = tf.placeholder(tf.int32,shape=(batchSize))

        # Use for test sets
        testy = tf.placeholder(tf.int32, [None, ])
        testx = tf.placeholder(tf.float32, [None, 1764])

        image_batch, label_batch = input_pipeline(os.path.join(data_folder,"train"), batchSize, num_epochs=None)

        W0, b0 = initWeightBias(x.shape[1], 2560)
        layer1 = layer(x, W0, b0)

        W1, b1 = initWeightBias(2560, 4)
        pred = layer(layer1, W1, b1)

        lam = 0.00000
        decay_penalty = tf.add(tf.mul(lam, tf.reduce_sum(tf.square(W0))), tf.mul(lam, tf.reduce_sum(tf.square(W1))))
        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y_)
        cost = tf.add(tf.reduce_mean(entropy), decay_penalty)

        train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(cost, global_step=globalStep)

        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.cast(testy, tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init = tf.initialize_all_variables()

        sess.run(init)

        test_x, test_y = input_pipeline(os.path.join(data_folder,"test"), batchSize, num_epochs=1)

        timers = []
        time_train = []
        time_run = []

        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            steps = 0
            while not coord.should_stop():

                now = time.time()

                sess.run(train_step, feed_dict={x: image_batch, y_: label_batch},)
                time_run.append(time.time() - now)
                steps += 1
                print(globalStep)
                if steps % 1000 == 0:
                    # print(batch_y)
                    #print("steps=" + str(steps))
                    # for i in range(10):
                    #temp = sess.run(accuracy, feed_dict={x: image_batch, y_: label_batch})

                    print "Test:", sess.run(accuracy, feed_dict={x: test_x, testy: test_y})
                    print "Train:", sess.run(accuracy, feed_dict={x: image_batch, testy: label_batch})
                    #print "Penalty:", sess.run(decay_penalty)


                    # snapshot = {}
                    # snapshot["W0"] = sess.run(W0)
                    # snapshot["W1"] = sess.run(W1)
                    # snapshot["b0"] = sess.run(b0)
                    # snapshot["b1"] = sess.run(b1)
                    # cPickle.dump(snapshot,  open("new_snapshot"+str(i)+".pkl", "w"))
                timers.append(time.time() - now)
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        coord.join(threads)

        sess.close()


if __name__=="__main__":
    data_folder = "/home/ujash/nvme/data2"
    hog1layer(1000, .0001, data_folder)
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