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
                                    snapshot=snapshot)


def vgg_model1(batch_size, snapshot=None, global_step=None):
    learning_rate = .00001
    imgs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name="Inputs")
    y_ = tf.placeholder(tf.int32, shape=(batch_size), name="Outputs")
    return vgg16.VGG16(imgs,y_, learning_rate, max_pool_num=5, global_step=global_step, snapshot=snapshot)


def run_model(model, sess, global_step, read_func):
    timers = {"batching": 0., "converting": 0., "training": 0., "testing": 0., "acc": 0., "total_tests": 0.}
    test_steps = 1000
    valid_steps = 10000
    snapshot = {}
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    steps = 0
    try:
        print("Starting training")
        while not coord.should_stop():

            now = time.time()
            imgs, labels = sess.run([image_batch, label_batch])
            timers["batching"] += (time.time() - now)

            now = time.time()
            imgs = read_func(imgs)
            timers["converting"] += (time.time() - now)

            steps += 1
            # if steps % test_steps == 0:
            #     print(sess.run(global_step))
            #     print("Train: " + str(sess.run(model.acc, feed_dict={model.inputs: imgs, model.testy: labels})))

            now = time.time()
            sess.run(model.train_step, feed_dict={model.inputs: imgs, model.labels: labels})
            timers["training"] += (time.time() - now)

            # print(sess.run(model.global_step))

            if steps % test_steps == 0:
                print(steps)
                print("Calculating test accuracy")
                test_acc, test_time = run_acc_batch(num_test_images, test_images, test_labels, model, sess,
                                                    max_parallel_calcs=max_parallel_acc_calcs)
                timers["testing"] += test_time
                timers["total_tests"] += 1
                print("Test: " + str(test_acc))

                if steps%valid_steps == 0:
                    print("Calculating validation accuracy")
                    acc_valid, valid_time = run_acc_batch(num_valid_images, valid_images, valid_labels, model, sess,
                                                          max_parallel_calcs=max_parallel_acc_calcs)

                    print("Valid: " + str(acc_valid))

                    for key in model.parameters.keys():
                        snapshot[key] = sess.run(model.parameters[key])
                    print(timers)
                    pickle.dump(snapshot, open(os.path.join(data_folder, snapshot_folder, str(steps // valid_steps)
                                                            + ".pkl"), "wb"))

                if test_acc >.99:
                    break
                print("Training")
                # snapshot = {}
            # timers.append(time.time() - now)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    coord.join(threads)

    for key in model.parameters.keys():
        snapshot[key] = sess.run(model.parameters[key])

    pickle.dump(snapshot,
                open(os.path.join(data_folder, snapshot_folder, "Final.pkl"), "wb"))

    sess.close()
    for timer in timers.keys():
        if timer.count("acc") == 0:
            print(timer + " avg: " + str(timers[timer]/steps))
    print("acc avg: " + str(timers["acc"]/timers["total_tests"]))


def run_acc_batch(num_images, images, labels, model, sess, max_parallel_calcs=None):
    acc = 0.
    total_test = 0
    now = time.time()
    repeat_num = 1
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    if max_parallel_calcs:
        repeat_num = num_images // max_parallel_calcs
    print(repeat_num)
    try:
        while not coord.should_stop():
            for i in range(repeat_num):
                raw_imgs_list, labels_list = sess.run([images, labels])
                imgs_list = read_func(raw_imgs_list)
                total_test += len(imgs_list)
                acc += sess.run(model.acc, feed_dict={model.inputs: imgs_list, model.testy: labels_list,
                                                      model.keep_probs: 1})
            break
    finally:
        coord.request_stop()
    coord.join(threads)
    timer = (time.time() - now)
    print(acc)
    return acc/repeat_num, timer


def split_test_by_tags():
    test_folder = "C:\\PhotoOrientation\\data\\data2\\test\\images"
    dirs = [d for d in os.listdir(test_folder) if os.path.isdir(os.path.join(test_folder, d))]
    print(dirs)
    num_images = 0
    for dir in dirs:
        label = int(dir)//90
        print(label)
        for root_inner, dirnames_inner, filenames_actual in os.walk(os.path.join(test_folder, dir)):
            for filename in filenames_actual:
                print(dirnames_inner)
                print(os.path.join(root_inner, filename))
                print(label)
                num_images += 1
                if num_images == 10:
                    break

if __name__ == "__main__":
    cur_model = None
    read_func = dummy_reader
    feature = "images"

    data_folder = os.path.join("C:", os.sep, "PhotoOrientation", "data", "data2")
    print(data_folder)
    globalStep = tf.Variable(0, name='global_step', trainable=False)
    ses = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    vgg = True
    if vgg:
        batch_size = 25
        max_parallel_acc_calcs = 25
        snapshot_folder = "snapshotVGG"
        snapshot_file = "5.pkl"
        if os.path.exists(data_folder):
            M = pickle.load(open(os.path.join(data_folder, snapshot_folder, snapshot_file), 'rb'))
            print("Snapshot Loaded")
        else:
            M = np.load('vgg16_weights.npz')
        feature = "images"
        snapshot_folder = "snapshotVGG1"
        cur_model = vgg_model1(batch_size, snapshot=M, global_step=globalStep)
    else:
        batch_size = 100
        max_parallel_acc_calcs = 1000
        snapshot_folder = "snapshotHOG"
        snapshot_file = "457.pkl"
        if os.path.exists(data_folder):
            M = pickle.load(open(os.path.join(data_folder, snapshot_folder, snapshot_file), 'rb'))
            print("Snapshot Loaded")
        else:
            M = pickle.load(open("snapshotHOG\\457.pkl", 'rb'))

        cur_model = hog_model(batch_size, snapshot=M, global_step=globalStep)

        feature="hog2"
        read_func = convert_binary_to_array

    if not os.path.exists(os.path.join(data_folder, snapshot_folder)):
        os.mkdir(os.path.join(data_folder, snapshot_folder))

    num_test_images = 12000
    num_valid_images = 12000
    image_batch, label_batch = input_pipeline(data_folder, batch_size, data_set="train", feature=feature)
    test_images, test_labels = input_pipeline(data_folder, max_parallel_acc_calcs, data_set="test", feature=feature,
                                              num_images=num_test_images)
    valid_images, valid_labels = input_pipeline(data_folder, max_parallel_acc_calcs, data_set="valid", feature=feature,
                                                num_images=num_valid_images)
    init = tf.initialize_all_variables()
    ses.run(init)
    # if cur_model:
    #     run_model(cur_model, ses, globalStep, read_func)
    split_test_by_tags()

