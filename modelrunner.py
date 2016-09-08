import numpy as np
import vgg16
import neuralnet
import alexnet
import pickle
import os
import time
import tensorflow as tf
from datahandler import input_pipeline, read_file_format, create_labeled_image_list, convert_binary_to_array


def hog1layer(batch_size, snapshot=None, global_step=None):
    hid_layers = [(2560, tf.nn.tanh)]
    learning_rate = 0.00001
    x_width = 1764
    y_width = 4
    return neuralnet.LayeredNetwork(batch_size, x_width, y_width, hid_layers, learning_rate, global_step=global_step,
                                    snapshot = snapshot)


def run_model(model, sess):
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
                for i in range(12):
                    imgs_test, labels_test = sess.run([test_images, test_labels])
                    imgs_test = convert_binary_to_array(imgs_test)
                    total_test += len(imgs_test)
                    acc += sess.run(model.acc, feed_dict={model.inputs: imgs_test, model.testy: labels_test})
                timers["testing"] += (time.time() - now)
                timers["total_tests"] += 1

                print("Test: " + str(acc/12))

                if steps%10000 == 0:
                    acc_valid = 0.
                    for i in range(12):
                        imgs_valid, labels_valid = sess.run([valid_images, valid_labels])
                        imgs_valid = convert_binary_to_array(imgs_valid)
                        acc_valid += sess.run(model.acc,
                                              feed_dict={model.inputs: imgs_valid, model.testy: labels_valid})

                    print("Valid: " + str(acc_valid/12))

                    for key in model.parameters.keys():
                        snapshot[key] = sess.run(model.parameters[key])

                    pickle.dump(snapshot,
                            open(os.path.join(data_folder, "snapshot1H" + str(steps // 10000) + ".pkl"), "wb"))
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
    batch_size = 1000

    data_folder = "/home/ujash/nvme/data2"
    snapshot_filename = "snapshot1000H25.pkl"
    tester= np.load("vgg16_weights.npz")
    print(tester.keys())
    '''
    if os.path.exists(data_folder):
        M = pickle.load(open(os.path.join(data_folder,snapshot_filename),'rb'))
        print("Snapshot Loaded")
    else:
        M = pickle.load(open("snapshotHOG458.pkl",'rb'))

    keys = sorted(M.keys())
    weights = {}
    weights["w0"] = M["w0"]
    weights["b0"] = M["b0"]

    weights["wOutput"] = M["w1"]
    weights["bOutput"] = M["b1"]

    global_step = tf.Variable(0, name='global_step', trainable=False)
    ses = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    hog_net = hog1layer(batch_size, snapshot=weights, global_step = global_step)

    feature="hog2"
    image_batch, label_batch = input_pipeline(data_folder, batch_size, data_set="train", feature=feature)
    test_images, test_labels = input_pipeline(data_folder, 1000, data_set="test", feature=feature, num_images=12000)
    valid_images, valid_labels = input_pipeline(data_folder, 1000, data_set="valid", feature=feature, num_images=12000)

    init = tf.initialize_all_variables()
    ses.run(init)

    run_model(hog_net, ses)
    '''
