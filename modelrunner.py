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


def vgg_model1(batch_size, fc_size, snapshot=None, global_step=None):
    learning_rate = .00001
    imgs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name="Inputs")
    y_ = tf.placeholder(tf.int32, shape=(batch_size), name="Outputs")
    return vgg16.VGG16(imgs,y_, learning_rate, fc_size=fc_size, max_pool_num=5,
                       global_step=global_step, snapshot=snapshot)


def run_model(model, sess, global_step, read_func, data_folder, snapshot_folder):
    timers = {"batching": 0., "converting": 0., "training": 0., "testing": 0., "acc": 0., "total_tests": 0.}
    test_steps = 10000
    valid_steps = 25000
    snapshot = {}
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    steps = 0
    try:
        print("Starting training")
        while not coord.should_stop():

            now = time.time()
            imgs, labels, tags = sess.run([image_batch, label_batch, tags_batch])
            timers["batching"] += (time.time() - now)

            now = time.time()
            imgs = read_func(imgs)
            timers["converting"] += (time.time() - now)

            # if steps % test_steps == 0:
            #     print(sess.run(global_step))
            #     print("Train: " + str(sess.run(model.acc, feed_dict={model.inputs: imgs, model.testy: labels})))

            now = time.time()
            sess.run(model.train_step, feed_dict={model.inputs: imgs, model.labels: labels})
            timers["training"] += (time.time() - now)

            # print(sess.run(model.global_step))
            if steps % 2000 == 0:
                print("Step: " + str(steps))
            if steps % test_steps == 0:
                print(steps)
                print("Calculating test accuracy")
                test_acc, test_time = run_acc_batch(num_test_images, test_images, test_labels, test_tags, model, sess,
                                                    max_parallel_calcs=max_parallel_acc_calcs)
                timers["testing"] += test_time
                timers["total_tests"] += 1
                print("Test: " + str(test_acc))
                if test_acc > .99:
                    print("Achieved very high accuracy, stopping")
                    break
                if steps % valid_steps != 0:
                    print("Resume training")
            steps += 1

            if steps % valid_steps == 0:
                print("Calculating validation accuracy")
                acc_valid, valid_time = run_acc_batch(num_valid_images, valid_images, valid_labels, valid_tags,
                                                      model, sess, max_parallel_calcs=max_parallel_acc_calcs)

                print("Valid: " + str(acc_valid))

                for key in model.parameters.keys():
                    snapshot[key] = sess.run(model.parameters[key])
                print(timers)
                pickle.dump(snapshot, open(os.path.join(snapshot_folder, str(steps // valid_steps)
                                                        + ".pkl"), "wb"))

                print("Resume training")
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
                open(os.path.join(snapshot_folder, "Final.pkl"), "wb"))

    sess.close()
    for timer in timers.keys():
        if timer.count("acc") == 0:
            print(timer + " avg: " + str(timers[timer]/steps))
    print("acc avg: " + str(timers["acc"]/timers["total_tests"]))


def run_acc_batch(num_images, images, labels, tags, model, sess, max_parallel_calcs=None):
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
                raw_imgs_list, labels_list, tags_list = sess.run([images, labels, tags])
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


def parallel_acc_by_tags(model, sess, max_parallel_calcs, data_folder, data_set="test", feature="images"):
    if data_set:
        data_folder = os.path.join(data_folder, data_set)
    if feature:
        data_folder = os.path.join(data_folder, feature)
    orientations = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]
    print(orientations)
    total_images = 0
    images, labels, tags = input_pipeline(data_folder_loc, max_parallel_calcs, data_set=None, feature=feature,
                                          binary_file=bin_or_not, num_epochs=1, num_images=None)

    incorrect_images_list = tf.Variable([], dtype=tf.string, trainable=False, name="Incorrect_images")
    adder_image_names = tf.placeholder(dtype=tf.string, shape=[None], name="Adder_images")
    new_incorrect_images_list = tf.concat(0, [incorrect_images_list, adder_image_names])
    add_incorrect_images = tf.assign(incorrect_images_list, new_incorrect_images_list, use_locking=True,
                                     validate_shape=False)

    init_ops = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    sess.run(init_ops)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    steps = 0
    try:
        print("Checking Accuracy")
        while not coord.should_stop():
            steps += 1
            raw_imgs_list, labels_list, tags_list = sess.run([images, labels, tags])
            imgs_list = read_func(raw_imgs_list)
            preds = sess.run(model.correct_predictions, feed_dict={model.inputs: imgs_list, model.testy: labels_list,
                                                                     model.keep_probs: 1})
            total_images += len(preds)
            incorrect_indices = np.where(preds == 0)

            # Uses locking so we do not lose any incorrect classifications
            sess.run(add_incorrect_images, feed_dict={adder_image_names: tags_list[incorrect_indices]})
            if steps % 100 == 0:
                print("Calculated " + str(steps*max_parallel_calcs) + " files")
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    coord.join(threads)
    inc_name = sess.run(incorrect_images_list)
    print("Correct classifications: " + str(total_images - len(inc_name)))
    print("Total images: " + str(total_images))
    print("Accuracy: " + str((total_images - len(inc_name))/total_images))
    with open(os.path.join(data_folder, "incorrect.txt"), 'w') as f:
        for each in inc_name:
            f.write(os.path.join(data_folder, each.decode('utf-8'))+'\n')
    sess.close()


def split_acc_by_tags(model, sess, data_folder, snapshot_filename, data_set="test", feature="images"):
    stat_filename = os.path.split(os.path.split(snapshot_filename)[0])[1] + "-" + \
                    os.path.split(snapshot_filename)[1][:-4] + "-" + \
                    data_set
    if data_set:
        data_set = os.path.join(data_folder, data_set)
    else:
        data_set = data_folder
    if feature:
        feature = os.path.join(data_set, feature)
    else:
        feature = data_set
    orientations = [d for d in os.listdir(feature) if os.path.isdir(os.path.join(feature, d))]
    print(orientations)
    image_stats = {}
    total_images = 0
    correct_images = 0

    for orientation in orientations:
        label = int(orientation)//90
        orientation_dir = os.path.join(feature, orientation)
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
                        with open(os.path.join(data_folder, "stats", stat_filename + ".txt"), "a") as incorrect_stored:
                            incorrect_stored.write(os.path.join(cur_dir, image_file)+"\n")
            print(image_stats[tag])

    print("Correct: " + str(correct_images) + "\nTotal: " + str(total_images))
    print("Total acc: ", str(correct_images/total_images))

    with open(os.path.join(data_folder, "stats", stat_filename + ".csv"), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        writer.writerow(["Tag", "Landscape Correct", "Landscape Total", "Portrait Correct", "Portrait Total"])
        for key, value in image_stats.items():
            writer.writerow([key, value["L_correct"], value["L_total"], value["P_correct"], value["P_total"]])

    sess.close()

# Test by category outside of these ones
# Take average of each category
# Train by category and test same category
# Abstract art
# Test with different bounding boxes

if __name__ == "__main__":
    cur_model = None
    read_func = dummy_reader
    feature_type = "images"
    data = "set1"
    # data_folder_loc = os.path.join("C:", os.sep, "PhotoOrientation", "SUN397", data)
    data_loc = os.getenv('data_loc')
    # print(data_folder_loc)
    for filename in ["test.txt", "train.txt", "valid.txt"]:
        with open(os.path.join(os.getcwd(), "datasets", data, feature_type, filename), 'r', newline='\n') as f:
            text = f.read().split('\r\n')
            print(len(text))
            if len(text) == 1:
                text = text[0].split('\n')
            print(len(text))
            saved = []
            for each in text:
                if each != '':
                    saved.append(os.path.join(data_loc, feature_type, each.split(feature_type)[1][1:]))
        if not os.path.exists(os.path.join(os.getcwd(), "temp", feature_type)):
            os.makedirs(os.path.join(os.getcwd(), "temp", feature_type))
        with open(os.path.join(os.getcwd(), "temp", feature_type, filename), 'w', newline='\n') as f:
            for each in saved:
                f.write(each + '\n')
    data_folder_loc = os.path.join(os.getcwd(), "temp")

    globalStep = tf.Variable(0, name='global_step', trainable=False)
    ses = tf.Session()  # config=tf.ConfigProto(log_device_placement=True))

    vgg = True
    load_snapshot_filename = "C:\\PhotoOrientation\\data\\SUN397\\snapshotVGG3\\2.pkl"
    snapshot_save_folder = "C:\\PhotoOrientation\\data\\SUN397\\snapshotVGG4"
    if vgg:
        batch_size = 10
        max_parallel_acc_calcs = 20

        if os.path.exists(load_snapshot_filename):
            M = pickle.load(open(load_snapshot_filename, 'rb'))
            print("Snapshot Loaded")
            Z = {}
            for each in M:
                Z[each] = M[each]

        else:
            print("Snapshot not found, loading default weights")
            M = np.load('vgg16_weights.npz')
            # Change last to 4 layers
            Z = {}
            for each in M:
                Z[each] = M[each]
        if M['fc8_W'].shape[1] != 4:
            Z['fc8_W'] = M['fc8_W'][:, :4]
            Z['fc8_b'] = M['fc8_b'][:4]
        feature_type = "images"
        cur_model = vgg_model1(batch_size, fc_size=4096, snapshot=Z, global_step=globalStep)
        bin_or_not = False
    else:
        batch_size = 100
        max_parallel_acc_calcs = 1000
        if os.path.exists(load_snapshot_filename):
            M = pickle.load(open(os.path.join(load_snapshot_filename), 'rb'))
            print("Snapshot Loaded")
        else:
            print("Snapshot not found, loading default weights")
            M = pickle.load(open("snapshotHOG\\457.pkl", 'rb'))

        cur_model = hog_model(batch_size, snapshot=M, global_step=globalStep)

        feature_type = "hog2"
        read_func = convert_binary_to_array
        bin_or_not = True

    # parallel_acc_by_tags(cur_model, ses, max_parallel_acc_calcs, data_folder_loc, data_set="", feature="images")
    # exit()

    # split_acc_by_tags(cur_model, ses, data_folder, load_snapshot_filename, data_set="train", feature="images")
    # exit()

    num_test_images = (17276 // batch_size) * batch_size
    num_valid_images = (21596 // batch_size) * batch_size
    training_epochs = 1
    with tf.device("/cpu:0"):
        image_batch, label_batch, tags_batch = input_pipeline(data_folder_loc, batch_size, data_set="train",
                                                              feature=feature_type, binary_file=bin_or_not,
                                                              from_file=True,
                                                              num_epochs=6)
        test_images, test_labels, test_tags = input_pipeline(data_folder_loc, max_parallel_acc_calcs, data_set="test",
                                                             feature=feature_type, num_images=num_test_images,
                                                             binary_file=bin_or_not, orientations=[0], from_file=True)
        valid_images, valid_labels, valid_tags = input_pipeline(data_folder_loc, max_parallel_acc_calcs,
                                                                data_set="valid",
                                                                feature=feature_type, num_images=num_valid_images,
                                                                binary_file=bin_or_not, orientations=[0],
                                                                from_file=True)
    init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
    ses.run(init)

    if cur_model:
        if not os.path.exists(os.path.join(data_folder_loc, snapshot_save_folder)):
            os.makedirs(os.path.join(data_folder_loc, snapshot_save_folder))
        run_model(cur_model, ses, globalStep, read_func, data_folder_loc, snapshot_save_folder)

