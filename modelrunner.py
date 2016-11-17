import numpy as np
import vgg16
import neuralnet
import pickle
import os
import time
import tensorflow as tf
from datahandler import input_pipeline, convert_binary_to_array
from scipy.misc import imread, imresize
import csv


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


def split_acc_by_tags(model, sess, data_set="test"):
    data_folder = "C:\\PhotoOrientation\\data2\\" + data_set + "\\images"
    orientations = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]
    print(orientations)
    image_stats = {}
    total_images = 0
    correct_images = 0

    for orientation in orientations:
        label = int(orientation)//90
        orientation_dir = os.path.join(data_folder, orientation)
        tags = [d for d in os.listdir(orientation_dir) if os.path.isdir(os.path.join(orientation_dir, d))]
        for tag in tags:
            print(tag)
            if tag not in image_stats:
                image_stats[tag] = {}
            tag_dir = os.path.join(orientation_dir, tag)
            layouts = [d for d in os.listdir(tag_dir) if os.path.isdir(os.path.join(tag_dir, d))]

            for layout in layouts:
                if (layout+"_total") not in image_stats[tag]:
                    image_stats[tag][layout+"_total"] = 0
                    image_stats[tag][layout+"_correct"] = 0
                cur_dir = os.path.join(tag_dir, layout)
                image_files = os.listdir(cur_dir)

                for image_file in image_files:
                    loaded_image = imread(os.path.join(cur_dir,image_file), mode='RGB')
                    loaded_image = imresize(loaded_image, (224, 224))
                    acc = sess.run(model.acc, feed_dict={model.inputs: [loaded_image], model.testy: [label],
                                                         model.keep_probs: 1})
                    total_images += 1
                    image_stats[tag][layout + "_total"] += 1

                    if acc > .5:
                        correct_images += 1
                        image_stats[tag][layout + "_correct"] += 1
                    else:
                        with open(os.path.join("C:\\PhotoOrientation\\data2\\stats", snapshot_folder + "-" +
                                  snapshot_file[:-4] + "-" + data_set + ".txt"), "a") as incorrect_stored:
                            incorrect_stored.write(os.path.join(cur_dir, image_file)+"\n")
            print(image_stats[tag])

    print("Correct: " + str(correct_images) + "\nTotal: " + str(total_images))
    print("Total acc: ", str(correct_images/total_images))

    with open(os.path.join("C:\\PhotoOrientation\\data2\\stats", snapshot_folder + "-" + snapshot_file[:-4] + "-" +
              data_set + ".csv"), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        writer.writerow(["Tag", "Landscape Correct", "Landscape Total", "Portrait Correct", "Portrait Total"])
        for key, value in image_stats.items():
            writer.writerow([key, value["L_correct"], value["L_total"], value["P_correct"], value["P_total"]])

    sess.close()

# Test by category outside of these ones
# Take average of each category
# Train by category and test same category

if __name__ == "__main__":

    cur_model = None
    read_func = dummy_reader
    feature = "images"

    data_folder = os.path.join("C:", os.sep, "PhotoOrientation", "data2")
    print(data_folder)
    globalStep = tf.Variable(0, name='global_step', trainable=False)
    ses = tf.Session()#config=tf.ConfigProto(log_device_placement=True))

    vgg = True
    if vgg:
        batch_size = 25
        max_parallel_acc_calcs = 25
        snapshot_folder = "snapshotVGG1"
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

    init = tf.initialize_all_variables()
    ses.run(init)
    split_acc_by_tags(cur_model, ses, "train")
    exit()

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

