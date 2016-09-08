import numpy as np
import vgg16
import neuralnet
import pickle
import os
import time
import tensorflow as tf
from datahandler import input_pipeline, convert_binary_to_array


def dummy_reader(input_data):
    return input_data


def hog_model(batch_size, snapshot=None, global_step=None):
    hid_layers = [(2560, tf.nn.tanh)]
    learning_rate = 0.00001
    x_width = 1764
    y_width = 4
    return neuralnet.LayeredNetwork(batch_size, x_width, y_width, hid_layers, learning_rate, global_step=global_step,
                                    snapshot = snapshot)


def vgg_model1(batch_size, snapshot=None, global_step=None):
    learning_rate = .00001
    imgs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name="Inputs")
    y_ = tf.placeholder(tf.int32,shape=(batch_size), name="Outputs")
    temp_model = vgg16.VGG16(imgs, y_, learning_rate, global_step=globalStep, snapshot=M)

    #Changing last layer only
    with tf.name_scope('fc8') as scope:
        if snapshot and snapshot['fc8_b'].shape[0] == 4:
            wl = snapshot['fc8_W']
            bl = snapshot['fc8_b']
        else:
            wl = tf.truncated_normal([4096, 4], dtype=tf.float32)
            bl = tf.constant(1.0, shape=[4], dtype=tf.float32)
        temp_model.parameters["fc8_W"] = tf.Variable(wl, name='weights')
        temp_model.parameters["fc8_b"] = tf.Variable(bl, trainable=True, name='biases')
        temp_model.outputs = tf.nn.bias_add(tf.matmul(temp_model.tensors['fc7'],
                                                      temp_model.parameters['fc8_W']), temp_model.parameters['fc8_b'])

    temp_model.train_step = temp_model.training()
    return temp_model


def run_model(model, sess, global_step, read_func):
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
            imgs = read_func(imgs)
            timers["converting"] += (time.time() - now)

            steps += 1
            if steps % 1000 == 0:
                print(sess.run(global_step))
                print("Train: " + str(sess.run(model.acc, feed_dict={model.inputs: imgs, model.testy: labels})))

            now = time.time()
            sess.run(model.train_step, feed_dict={model.inputs: imgs, model.labels: labels})
            timers["training"] += (time.time() - now)


            #print(sess.run(model.global_step))
            if steps % 1000 == 0:
                acc = 0.
                total_test = 0
                now = time.time()
                for i in range(120):
                    imgs_test, labels_test = sess.run([test_images, test_labels])
                    imgs_test = read_func(imgs_test)
                    total_test += len(imgs_test)
                    acc += sess.run(model.acc, feed_dict={model.inputs: imgs_test, model.testy: labels_test})
                timers["testing"] += (time.time() - now)
                timers["total_tests"] += 1

                print("Test: " + str(acc/120))

                if steps%10000 == 0:
                    acc_valid = 0.
                    for i in range(120):
                        imgs_valid, labels_valid = sess.run([valid_images, valid_labels])
                        imgs_valid = read_func(imgs_valid)
                        acc_valid += sess.run(model.acc,
                                              feed_dict={model.inputs: imgs_valid, model.testy: labels_valid})

                    print("Valid: " + str(acc_valid/120))

                    for key in model.parameters.keys():
                        snapshot[key] = sess.run(model.parameters[key])

                    pickle.dump(snapshot,
                            open(os.path.join(data_folder, "snapshot1VGG" + str(steps // 10000) + ".pkl"), "wb"))
                if (acc/120)>.99:
                    break
                # snapshot = {}
            #timers.append(time.time() - now)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    coord.join(threads)

    for key in model.parameters.keys():
        snapshot[key] = sess.run(model.parameters[key])

    pickle.dump(snapshot,
                open(os.path.join(data_folder, "snapshotHOGFinal.pkl"), "wb"))

    sess.close()
    for timer in timers.keys():
        if timer.count("acc") == 0:
            print(timer + " avg: " + str(timers[timer]/steps))
    print("acc avg: " + str(timers["acc"]/timers["total_tests"]))


if __name__ == "__main__":
    cur_model=None
    read_func = dummy_reader
    feature="images"

    batch_size = 50
    data_folder = "/home/ujash/nvme/data2"

    globalStep = tf.Variable(0, name='global_step', trainable=False)
    ses = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    hog = False
    if hog:
        snapshot_filename = "snapshot1H2.pkl"
        if os.path.exists(data_folder):
            M = pickle.load(open(os.path.join(data_folder,snapshot_filename),'rb'))
            print("Snapshot Loaded")
        else:
            M = pickle.load(open("snapshotHOG458.pkl",'rb'))

        cur_model = hog_model(batch_size, snapshot=M, global_step = globalStep)

        feature="hog2"
        read_func = convert_binary_to_array



    vgg = True
    if vgg:
        M = np.load('vgg16_weights.npz')
        feature="images"
        cur_model = vgg_model1(batch_size, snapshot=M, global_step=globalStep)

    image_batch, label_batch = input_pipeline(data_folder, batch_size, data_set="train", feature=feature)
    test_images, test_labels = input_pipeline(data_folder, 100, data_set="test", feature=feature, num_images=12000)
    valid_images, valid_labels = input_pipeline(data_folder, 100, data_set="valid", feature=feature, num_images=12000)
    init = tf.initialize_all_variables()
    ses.run(init)
    if cur_model:
        run_model(cur_model, ses, globalStep,read_func)