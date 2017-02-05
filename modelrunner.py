import numpy as np
import vgg16
import neuralnet
import pickle
import os
import time
import tensorflow as tf
from datahandler import input_pipeline, convert_binary_to_array, get_dataset_mean
from scipy.misc import imread, imresize, imsave
import csv
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops


def dummy_reader(input_data):
    return input_data


def hog_model(num_images, snapshot=None, global_step=None):
    hid_layers = [(2560, tf.nn.tanh)]
    learning_rate = 0.00001
    x_width = 1764
    y_width = 4
    return neuralnet.LayeredNetwork(num_images, x_width, y_width, hid_layers, learning_rate, global_step=global_step,
                                    snapshot=snapshot)


def vgg_model1(batch_size, fc_size, snapshot=None, global_step=None, get_gradients=False, data_mean=None):
    learning_rate = .00001

    return vgg16.VGG16(batch_size, learning_rate, fc_size=fc_size, max_pool_num=5, guided_grad=get_gradients,
                       global_step=global_step, snapshot=snapshot, data_mean=data_mean)


def vgg_model2(batch_size, fc_size, snapshot=None, global_step=None, get_gradients=False, data_mean=None):
    learning_rate = .00001

    return vgg16.VGG16(batch_size, learning_rate, fc_size=fc_size, max_pool_num=4, guided_grad=get_gradients,
                       global_step=global_step, snapshot=snapshot, data_mean=data_mean)


def vgg_model(batch_size, fc_size=4096, max_pool_layers=5, snapshot=None, global_step=None, get_gradients=False,
              data_mean=None, pre_fc=False):
    learning_rate = .00001

    return vgg16.VGG16(batch_size, learning_rate, fc_size=fc_size, max_pool_num=max_pool_layers,
                       guided_grad=get_gradients,
                       global_step=global_step, snapshot=snapshot, data_mean=data_mean, pre_fc=pre_fc)


def run_model(model, sess, train_data, valid_data, test_data, batch_size, global_step, read_func, snapshot_folder,
              dropout=.75):
    timers = {"batching": 0., "converting": 0., "training": 0., "testing": 0., "acc": 0., "total_tests": 0.}

    # Steps at which to calculate test and valid
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
            imgs, labels, tags = sess.run([train_data['images'], train_data['labels'], train_data['tags']])
            timers["batching"] += (time.time() - now)

            now = time.time()
            imgs = read_func(imgs)
            timers["converting"] += (time.time() - now)

            # if steps % test_steps == 0:
            #     print(sess.run(global_step))
            #     print("Train: " + str(sess.run(model.acc, feed_dict={model.inputs: imgs, model.testy: labels})))

            now = time.time()
            sess.run(model.train_step1, feed_dict={model.inputs: imgs, model.labels: labels, model.keep_probs: dropout})
            timers["training"] += (time.time() - now)

            # print(sess.run(model.global_step))
            if steps % 1000 == 0:
                print("Step: " + str(steps))

            steps += 1

            if steps % test_steps == 0:
                print(steps)
                print("Calculating test accuracy")
                test_acc, test_time = run_acc_batch(test_data, model, sess, read_func,
                                                    max_parallel_calcs=batch_size*2)
                timers["testing"] += test_time
                timers["total_tests"] += 1
                print("Test: " + str(test_acc))
                if test_acc > .99:
                    print("Achieved very high accuracy, stopping")
                    break
                if steps % valid_steps != 0:
                    print("Resume training")

            if steps % valid_steps == 0:
                print("Calculating validation accuracy")
                acc_valid, valid_time = run_acc_batch(valid_data, model, sess, read_func,
                                                      max_parallel_calcs=batch_size*2)

                print("Valid: " + str(acc_valid))

                for key in model.parameters.keys():
                    snapshot[key] = sess.run(model.parameters[key])
                print(timers)
                snapshot['model_acc'] = acc_valid
                pickle.dump(snapshot, open(os.path.join(snapshot_folder, str(steps // valid_steps)
                                                        + ".pkl"), "wb"))

                print("Resume training")
            # snapshot = {}
            # timers.append(time.time() - now)
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        print("Calculating validation accuracy")
        acc_valid, valid_time = run_acc_batch(valid_data, read_func,
                                              model, sess, max_parallel_calcs=batch_size * 2)

        print("Valid: " + str(acc_valid))
        snapshot['model_acc'] = acc_valid

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


def run_acc_batch(data, model, sess, read_func, max_parallel_calcs=None):
    acc = 0.
    total_test = 0
    now = time.time()
    repeat_num = data['num_images']
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    if max_parallel_calcs:
        repeat_num = data['num_images'] // max_parallel_calcs
    print(repeat_num)
    try:
        while not coord.should_stop():
            for i in range(repeat_num):
                raw_imgs_list, labels_list, tags_list = sess.run([data['images'], data['labels'], data['tags']])
                imgs_list = read_func(raw_imgs_list)
                total_test += len(imgs_list)
                acc += sess.run(model.acc1, feed_dict={model.inputs: imgs_list, model.testy: labels_list,
                                                       model.keep_probs: 1})
            break
    finally:
        coord.request_stop()
    coord.join(threads)
    timer = (time.time() - now)
    print(acc)
    return acc/repeat_num, timer


def parallel_acc_by_tags(model, sess, max_parallel_calcs, data_folder, read_func, from_file=None, data_set="test",
                         feature="images", orientations=None):
    total_images = 0
    if orientations is None:
        orientations = [0, 90, 180, 270]
    images, labels, tags = input_pipeline(data_folder_loc, max_parallel_calcs, data_set=data_set,
                                          feature=feature, num_images=None,
                                          binary_file=False, orientations=orientations,
                                          from_file=from_file, num_epochs=1)

    incorrect_images_list = tf.Variable([], dtype=tf.string, trainable=False, name="Incorrect_images")
    adder_image_names = tf.placeholder(dtype=tf.string, shape=[None], name="Adder_images")
    new_incorrect_images_list = tf.concat(0, [incorrect_images_list, adder_image_names])
    add_incorrect_images = tf.assign(incorrect_images_list, new_incorrect_images_list, use_locking=True,
                                     validate_shape=False)

    incorrect_labels_list = tf.Variable([], dtype=tf.int32, trainable=False, name="Incorrect_image_labels")
    adder_image_labels = tf.placeholder(dtype=tf.int32, shape=[None], name="Adder_image_labels")
    new_incorrect_labels_list = tf.concat(0, [incorrect_labels_list, adder_image_labels])
    add_incorrect_labels = tf.assign(incorrect_labels_list, new_incorrect_labels_list, use_locking=True,
                                     validate_shape=False)

    init_ops = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
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
            sess.run(add_incorrect_labels, feed_dict={adder_image_labels: labels_list[incorrect_indices]})

            if steps % 100 == 0:
                print("Calculated " + str(steps*max_parallel_calcs) + " files")
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    coord.join(threads)
    inc_name = sess.run(incorrect_images_list)
    inc_label = sess.run(incorrect_labels_list)
    print("Correct classifications: " + str(total_images - len(inc_name)))
    print("Total images: " + str(total_images))
    print("Accuracy: " + str((total_images - len(inc_name))/total_images))
    with open(os.path.join(data_folder, "incorrect.txt"), 'w') as f:
        for i in range(len(inc_name)):
            f.write(os.path.join(data_folder, inc_name[i].decode('utf-8')) + ', ' + str(inc_label[i]*90) + '\n')
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


def get_gradient(sess, model, data, layers=None):
    if layers is None:
        layers = model.gradients.keys()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    steps = 0

    save_folder = "grad_desc_neg-pos_0"
    try:
        print("Getting gradients")
        while not coord.should_stop():
            imgs, labels, tags = sess.run([data['images'], data['labels'], data['tags']])
            steps += 1
            preds = sess.run(model.prediction, feed_dict={model.inputs: imgs, model.testy: labels, model.keep_probs: 1})
            image_tags = []
            image_labels = []
            image_tags.append(tags)
            image_labels.append(labels)
            for layer in layers:
                gradients = []

                gradient = sess.run(model.gradients[layer], feed_dict={model.inputs: imgs, model.testy: labels, model.keep_probs: 1})
                gradients.extend(gradient)

                for i in range(len(gradients)):
                    tester = np.sum(np.sum(np.sum(gradients[i], 1), 1), 1)
                    correct_indices = np.where(tester != 0.)[0]
                    for j in correct_indices: #range(len(gradient[i])):
                        # print(tags[i][j])
                        # print(image.dtype)
                        positive = np.array(gradient[i][j], copy=True)
                        negative = np.array(gradient[i][j], copy=True)
                        positive[np.where(gradient[i][j] < 0.)] = 0.
                        negative[np.where(gradient[i][j] > 0.)] = 0.
                        if preds[j] == image_labels[i][j]:
                            filepath = image_tags[i][j].decode('utf-8').replace(os.getenv('data_loc'),
                                                                                os.path.join(os.getcwd(),
                                                                                             "temp", save_folder,
                                                                                             "correct"))
                        else:
                            filepath = image_tags[i][j].decode('utf-8').replace(os.getenv('data_loc'),
                                                                                os.path.join(os.getcwd(),
                                                                                             "temp", save_folder,
                                                                                             "incorrect"))
                        if not os.path.exists(os.path.split(filepath)[0]):
                            os.makedirs(os.path.split(filepath)[0])
                        file_name = str(os.path.split(filepath)[1]).split(".")[0]
                        filepath = os.path.join(os.path.split(filepath)[0], file_name + "-" + layer + "-orient"
                                                + str(image_labels[i][j]) + "-pred" + str(preds[j]) + "-pos" + ".jpg")
                        imsave(filepath, positive, format='JPEG')
                        filepath = os.path.join(os.path.split(filepath)[0], file_name + "-" + layer + "-orient"
                                                + str(image_labels[i][j]) + "-pred" + str(preds[j]) + "-neg" + ".jpg")
                        imsave(filepath, negative, format='JPEG')
                        filepath = os.path.join(os.path.split(filepath)[0], file_name + "-" + layer + "-orient"
                                                + str(image_labels[i][j]) + "-pred" + str(preds[j]) + "-full" + ".jpg")
                        imsave(filepath, gradient[i][j], format='JPEG')
            if steps % 2000 == 0:
                print(steps)

    except tf.errors.OutOfRangeError:
        print('Done -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    coord.join(threads)
    sess.close()
    '''
        Generates a graph with the input images, classifications and gradients
    :param images: list of numpy arrays containing images, i.e. [dog, cat, spider]
                    in case of a single image, put it in brackets, i.e. [dog]
    :return:
            gradients and indices of the classes in caffe_classes
    '''
    return gradients, image_tags


def create_model_and_inputs(batch_size, acc_batch_size, snapshot_filename, num_images=None, train_epochs=None,
                            test_epochs=None, data_from_file=False, vgg=True, model_pools=5, data_mean=None,
                            get_gradients=False, pre_fc=False):
    model = None
    read_func = dummy_reader
    if vgg:
        feature_type = "images"
    else:
        feature_type = "hog"

    data = "set1"
    if num_images is None:
        num_test_images = ((21596 * 4) // images_batch_size) * images_batch_size
        num_valid_images = ((17276 * 4) // images_batch_size) * images_batch_size
    else:
        num_test_images = num_images
        num_valid_images = num_images

    data_loc = os.getenv('data_loc')
    # print(data_folder_loc)
    if data_from_file:
        temp_folder = os.path.join(os.getcwd(), "temp")
        for filename in ["test.txt", "train.txt", "valid.txt"]:
            with open(os.path.join(os.getcwd(), "datasets", data, feature_type, filename), 'r', newline='\n') as f:
                text = f.read().split('\r\n')
                if len(text) == 1:
                    text = text[0].split('\n')
                print(len(text))
                saved = []
                for each in text:
                    if each != '':
                        saved.append(os.path.join(data_loc, feature_type, each.split(feature_type)[1][1:]))
            if not os.path.exists(os.path.join(temp_folder, feature_type)):
                os.makedirs(os.path.join(temp_folder, feature_type))
            with open(os.path.join(temp_folder, feature_type, filename), 'w', newline='\n') as f:
                for each in saved:
                    f.write(each + '\n')
        data_loc = temp_folder

    globalStep = tf.Variable(0, name='global_step', trainable=False)
    sess = tf.Session()  # config=tf.ConfigProto(log_device_placement=True))

    if vgg:
        max_parallel_acc_calc = acc_batch_size

        if os.path.exists(snapshot_filename):
            M = pickle.load(open(snapshot_filename, 'rb'))
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
            print("resizing final weights")
            Z['fc8_W'] = M['fc8_W'][:, :4]
            Z['fc8_b'] = M['fc8_b'][:4]
        feature_type = "images"
        model = vgg_model(batch_size,
                          fc_size=4096,
                          max_pool_layers=model_pools,
                          get_gradients=get_gradients,
                          snapshot=Z,
                          global_step=globalStep,
                          data_mean=data_mean,
                          pre_fc=pre_fc)
        bin_or_not = False
    else:
        max_parallel_acc_calc = acc_batch_size
        if os.path.exists(snapshot_filename):
            M = pickle.load(open(os.path.join(snapshot_filename), 'rb'))
            print("Snapshot Loaded")
        else:
            print("Snapshot not found, loading default weights")
            M = pickle.load(open("snapshotHOG\\457.pkl", 'rb'))

        model = hog_model(batch_size, snapshot=M, global_step=globalStep)

        feature_type = "hog2"
        read_func = convert_binary_to_array
        bin_or_not = True

    with tf.device("/cpu:0"):
        train_images, train_labels, train_tags, train_num = input_pipeline(data_loc, batch_size, data_set="train",
                                                                           feature=feature_type, binary_file=bin_or_not,
                                                                           from_file=data_from_file,
                                                                           num_epochs=train_epochs)
        test_images, test_labels, test_tags, test_num = input_pipeline(data_folder_loc, max_parallel_acc_calc,
                                                                       data_set="test", feature=feature_type,
                                                                       num_images=num_test_images,
                                                                       binary_file=bin_or_not,
                                                                       orientations=[0, 90, 180, 270],
                                                                       from_file=data_from_file,
                                                                       num_epochs=test_epochs)
        valid_images, valid_labels, valid_tags, valid_num = input_pipeline(data_folder_loc, max_parallel_acc_calc,
                                                                           data_set="valid", feature=feature_type,
                                                                           num_images=num_valid_images,
                                                                           binary_file=bin_or_not,
                                                                           orientations=[0, 90, 180, 270],
                                                                           from_file=data_from_file,
                                                                           num_epochs=test_epochs)

    train_data = {'images': train_images, 'labels': train_labels, 'tags': train_tags, 'num_images': train_num}
    test_data = {'images': test_images, 'labels': test_labels, 'tags': test_tags, 'num_images': num_test_images}
    valid_data = {'images': valid_images, 'labels': valid_labels, 'tags': valid_tags, 'num_images': num_test_images}
    init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

    return sess, init, model, train_data, test_data, valid_data, read_func, globalStep


# Test by category outside of these ones
# Take average of each category
# Train by category and test same category
# Abstract art
# Test with different bounding boxes

if __name__ == "__main__":
    mean = None
    cur_model = False
    load_snapshot_filename = "C:\\PhotoOrientation\\data\\SUN397\\snapshotVGG3\\2.pkl"
    images_batch_size = 20
    snapshot_save_folder = "C:\\PhotoOrientation\\data\\SUN397\\snapshots\\VGGfcTrain"
    from_file = True
    gradient_desc = False

    training = not gradient_desc
    data_folder_loc = os.getenv('data_loc')
    max_acc_batch_size = 40
    # mean = get_dataset_mean(data_folder_loc)
    # mean = [92.3243125, 89.39240884, 82.58156112]

    if from_file:
        data_folder_loc = os.path.join(os.getcwd(), "temp")

    if gradient_desc:
        @ops.RegisterGradient("GuidedRelu")
        def _guided_relu_grad(op, grad):
            return tf.select(0. < grad, gen_nn_ops._relu_grad(grad, op.outputs[0]), tf.zeros(grad.get_shape()))
        gradient_layers = ["prob0", "prob1", "prob2", "prob3"]
        images_batch_size = 5
        max_acc_batch_size = images_batch_size
        with tf.Graph().as_default() as g:
            with g.gradient_override_map({'Relu': 'GuidedRelu'}):
                ses, initializer, cur_model, \
                    train, test, valid, data_reader, step = create_model_and_inputs(images_batch_size,
                                                                                    max_acc_batch_size,
                                                                                    load_snapshot_filename,
                                                                                    data_from_file=from_file,
                                                                                    vgg=True,
                                                                                    get_gradients=gradient_desc,
                                                                                    num_images=None,
                                                                                    model_pools=5,
                                                                                    test_epochs=1)
                ses.run(initializer)
                grads, tags = get_gradient(ses, cur_model, test, layers=gradient_layers)
                print(len(tags))
                print(len(grads))
                print(grads[0].shape)
                calc = [0., 0.]
                ses.close()
                print(calc[0], calc[1])
        exit()

    if training:
        ses, initializer, cur_model, \
            train, test, valid, data_reader, step = create_model_and_inputs(images_batch_size, max_acc_batch_size,
                                                                            load_snapshot_filename,
                                                                            data_from_file=from_file,
                                                                            vgg=True,
                                                                            get_gradients=gradient_desc,
                                                                            num_images=None, test_epochs=None,
                                                                            data_mean=mean,
                                                                            pre_fc=True)
        ses.run(initializer)
        if cur_model is not None:
            if not os.path.exists(snapshot_save_folder):
                os.makedirs(snapshot_save_folder)
            run_model(cur_model, ses, train, valid, test, images_batch_size, step, data_reader, snapshot_save_folder,
                      dropout=.7)
    '''
    # print("testing CorelDB")
    # data_folder_loc = os.path.join("C:", os.sep, "PhotoOrientation", "CorelDB")
    # parallel_acc_by_tags(cur_model, ses, images_batch_size*2, data_folder_loc,
    #                      from_file=False, data_set="", feature="images", orientations=[0, 90, 180, 270])
    # ses.close()
    # exit()
    '''

    ''' Create a file which saves incorrect image as '(image filename), (orientation)\n'
    # parallel_acc_by_tags(cur_model, ses, images_batch_size*2, data_folder_loc, data_set="", feature="images")
    # ses.close()
    # exit()
    '''

    ''' Create a file with stats for individual tags
    # split_acc_by_tags(cur_model, ses, data_folder, load_snapshot_filename, data_set="train", feature="images")
    # ses.close()
    # exit()
    '''

    ''' Calculate individual accuracy without starting training
    # acc_valid = 0.
    # acc_valid, valid_time = run_acc_batch(num_valid_images, valid['images'], valid['labels'], valid['tags'],
    #                                       cur_model, ses, max_parallel_calcs=images_batch_size * 2)
    # print("Valid: " + str(acc_valid))
    # ses.close()
    # exit()
    '''

    # Run Training


