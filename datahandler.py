import os
import math
import time
from random import shuffle
from scipy.misc import imread, imsave
from skimage.color import rgb2gray
import numpy as np
from skimage.feature import hog
from skimage import exposure
import pickle
from skimage.transform import rotate
from PIL import Image
import tensorflow as tf
'''
def read_my_file_format(filename_queue):
    reader = tf.WholeFileReader()
    key, record_string = reader.read(filename_queue[0])
    image = tf.image.decode_jpeg(record_string)
    label = filename_queue[1]
    return image, label
'''


def read_file_format(filename_queue, binary_file=False, rot_to_label=False):

    label = tf.cast(filename_queue[2], tf.int32)
    tags = filename_queue[1]

    if binary_file:
        image = tf.read_file(filename_queue[0])
    else:
        tf.Print(filename_queue[0], [filename_queue[1]])

        tensor_image = tf.read_file(filename_queue[0])

        image = tf.image.decode_jpeg(tensor_image, channels=3)

        multiplier = tf.div(tf.constant(224, tf.float32),
                            tf.cast(tf.maximum(tf.shape(image)[0], tf.shape(image)[1]), tf.float32))
        x = tf.cast(tf.round(tf.mul(tf.cast(tf.shape(image)[0], tf.float32), multiplier)), tf.int32)
        y = tf.cast(tf.round(tf.mul(tf.cast(tf.shape(image)[1], tf.float32), multiplier)), tf.int32)
        image = tf.image.resize_images(image, [x, y])

        if rot_to_label:
            image = tf.image.rot90(image, k=label)

        image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
    return image, label, tags


def input_pipeline(directory, batch_size, data_set="train", feature="images", orientations=None,
                   binary_file=False, num_epochs=None, num_images=None, labeled_data=False, rand_seed=None,
                   num_threads=10, from_file=False):
    with tf.name_scope('InputPipeline'):
        # Reads paths of images together with their labels
        if not rand_seed:
            rand_seed = int(time.time())
        if from_file:
            image_list, label_list, tags_list = create_labeled_image_list_from_file(directory, data_set=data_set,
                                                                                    feature=feature,
                                                                                    orientations=orientations,
                                                                                    num_images=num_images,
                                                                                    labeled_data=labeled_data)
            print("Loading images from file: " + str(len(image_list)))
            print(repr(image_list[0]))
            if os.path.exists(image_list[0]):
                print("interesting")
        else:
            image_list, label_list, tags_list = create_labeled_image_list(directory, data_set=data_set, feature=feature,
                                                                          orientations=orientations,
                                                                          num_images=num_images,
                                                                          labeled_data=labeled_data)
        input_queue = tf.train.slice_input_producer([image_list, tags_list, label_list], num_epochs=num_epochs)

        # Makes an input queue
        print("Created queue")
        image, label, tags = read_file_format(input_queue, binary_file=binary_file, rot_to_label=not labeled_data)
        print("Read all files")
        # min_after_dequeue defines how big a buffer we will randomly sample
        #   from -- bigger means better shuffling but slower start up and more
        #   memory used.
        # capacity must be larger than min_after_dequeue and the amount larger
        #   determines the maximum we will prefetch.  Recommendation:
        #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
        min_after_dequeue = 10000
        capacity = min_after_dequeue + 3 * batch_size
        if data_set and data_set.count("train") == 1:
            image_batch, label_batch, tags_batch = tf.train.shuffle_batch([image, label, tags], batch_size=batch_size,
                                                                          capacity=capacity,
                                                                          min_after_dequeue=min_after_dequeue,
                                                                          num_threads=num_threads, seed=rand_seed)
        else:
            print("Batching")
            image_batch, label_batch, tags_batch = tf.train.batch([image, label, tags], batch_size=batch_size,
                                                                  capacity=capacity, num_threads=num_threads)
            print("Finished batching")
        return image_batch, label_batch, tags_batch


def convert_binary_to_array(image_binary_list):
    images = []
    # print(image_binary_list)
    for image_binary in image_binary_list:
        images.append(pickle.loads(image_binary))
    return images


def convert_to_array(image_list):
    images = []
    # print(image_binary_list)
    for image in image_list:
        images.append(imread(image))
    return images


def create_labeled_image_list_from_file(directory, data_set, feature="images", orientations=None, num_images=None,
                                        labeled_data=False):
    if orientations is None:
        orientations = [0, 90, 180, 270]
    image_list = []
    label_list = []
    tags_list = []
    if feature is not None:
        directory = os.path.join(directory, feature)

    filename = os.path.join(directory, data_set + ".txt")
    for orientation in orientations:
        if labeled_data:
            filename = os.path.join(directory, str(orientation), data_set + ".txt")

        with open(filename, 'r', newline='\n') as f:
            temp_list = f.read().split('\n')
            i = -1
            while temp_list[i] == "":
                i -= 1
            if i != -1:
                print("Pruning " + str((i+1) * -1) + " entries")
                temp_list = temp_list[:i+1]

        temp_label_list = [orientation/90]*len(temp_list)
        image_list.extend(temp_list)
        tags_list.extend(temp_list)
        label_list.extend(temp_label_list)
    if num_images is not None:
        image_list = image_list[:num_images]
        label_list = label_list[:num_images]
        tags_list = tags_list[:num_images]
    return image_list, label_list, tags_list


def create_labeled_image_list(directory, data_set=None, feature="images",
                              orientations=None, num_images=None, labeled_data=False):
    if orientations is None:
        orientations = [0, 90, 180, 270]
    image_list = []
    label_list = []
    tags_list = []
    if data_set is not None:
        directory = os.path.join(directory, data_set)
    if feature is not None:
        directory = os.path.join(directory, feature)
    directory = os.path.normpath(os.path.expandvars(os.path.expanduser(directory)))
    if not os.path.exists(directory):
        print("Feature or data set does not exist")
        print(directory)
    for root, dirnames, filenames in os.walk(directory):
        print("Loading images from: " + root)
        if not labeled_data:
            for orientation in orientations:
                temp_images, temp_labels, temp_tags, num_images_found = \
                    create_labeled_image_list_helper(root, label=orientation/90, labeled_data=labeled_data,
                                                     limit=num_images)
                image_list.extend(temp_images)
                label_list.extend(temp_labels)
                tags_list.extend(temp_tags)
        else:
            for dirname in dirnames:
                if int(dirname) in orientations:
                    label = int(dirname)/90
                    temp_images, temp_labels, temp_tags, num_images_found = \
                        create_labeled_image_list_helper(os.path.join(root, dirname), label=label,
                                                         labeled_data=labeled_data, limit=num_images)
                    if num_images:
                        num_images -= num_images_found
                    image_list.extend(temp_images)
                    label_list.extend(temp_labels)
                    tags_list.extend(temp_tags)
        break  # Only checking the orientation directories here

    print(str(len(image_list)) + " images loaded")
    return image_list, label_list, tags_list


def create_labeled_image_list_helper(directory, label=None, labeled_data=False, limit=None):
    image_list = []
    label_list = []
    tags_list = []
    num_images = 0
    # Make sure we are working with absolute paths and not relative ones, otherwise .replace() will not work
    directory = os.path.normpath(os.path.expandvars(os.path.expanduser(directory)))
    for root, dirnames, filenames in os.walk(os.path.join(directory)):
        for filename in filenames:
            if limit is not None:
                if limit <= num_images:
                    break
            tags = filename
            res = root.replace(directory + os.path.sep, "")
            tags = os.path.join(res, tags)
            if labeled_data:  # If image has a label, get the directory of the label
                tags = os.path.join(os.path.basename(directory), tags)
            label_list.append(label)
            image_list.append(os.path.join(root, filename))
            tags_list.append(tags)
            num_images += 1
    return image_list, label_list, tags_list, num_images


def save_pickle(out_dir, tag_name, count_total, img_train_dict, hog_train_dict, img_test_dict, hog_test_dict,
                img_valid_dict, hog_valid_dict):
    for key in img_train_dict.keys():
        img_train_dict[key] = np.asarray(img_train_dict[key], dtype=np.float32)
        hog_train_dict[key] = np.asarray(hog_train_dict[key], dtype=np.float32)
        img_test_dict[key] = np.asarray(img_test_dict[key], dtype=np.float32)
        hog_test_dict[key] = np.asarray(hog_test_dict[key], dtype=np.float32)
        img_valid_dict[key] = np.asarray(img_valid_dict[key], dtype=np.float32)
        hog_valid_dict[key] = np.asarray(hog_valid_dict[key], dtype=np.float32)

    print("Saving " + tag_name)
    with open(os.path.join(out_dir,"train", "images", tag_name+".pkl"), "wb") as img_train_file:
        pickle.dump(img_train_dict, img_train_file, protocol=2)
    with open(os.path.join(out_dir, "train", "hog", tag_name + ".pkl"), "wb") as hog_train_file:
        pickle.dump(hog_train_dict, hog_train_file, protocol=2)
    with open(os.path.join(out_dir, "test", "images", tag_name + ".pkl"), "wb") as img_test_file:
        pickle.dump(img_test_dict, img_test_file, protocol=2)
    with open(os.path.join(out_dir, "test", "hog", tag_name + ".pkl"), "wb") as hog_test_file:
        pickle.dump(hog_test_dict, hog_test_file, protocol=2)
    with open(os.path.join(out_dir, "valid", "images", tag_name + ".pkl"), "wb") as img_valid_file:
        pickle.dump(img_valid_dict, img_valid_file, protocol=2)
    with open(os.path.join(out_dir, "valid", "hog", tag_name + ".pkl"), "wb") as hog_valid_file:
        pickle.dump(hog_valid_dict, hog_valid_file, protocol=2)


def normalizeArray(imageArray, resizeDims = (224,224), RGBtoBW = False):
    dims = imageArray.size
    normArray = None

    if imageArray.mode != 'RGB':
        return normArray
    elif RGBtoBW:
        #TODO: Make image RGB
        pass

    # Cannot normalize if resizeDims are larger than the image
    if (dims[0] >= resizeDims[0]) and (dims[1] >= resizeDims[1]):
        normArray = imageArray.resize(resizeDims, Image.ANTIALIAS)

    return normArray


def make_valid(image, hog_fd, degrees, out_dir, dirname, dirname_inner, filename):
    image_out_path = os.path.join(out_dir, "valid", "images", degrees, dirname, dirname_inner)
    hog_out_path = os.path.join(out_dir, "valid", "hog", degrees, dirname, dirname_inner)
    if not os.path.exists(image_out_path):
        os.makedirs(image_out_path)
    if not os.path.exists(hog_out_path):
        os.makedirs(hog_out_path)
    np.save(os.path.join(hog_out_path, filename[:-4]), hog_fd)
    imsave(os.path.join(image_out_path, filename), image, format='JPEG')


def make_test(image, hog_fd, degrees, out_dir, dirname, dirname_inner, filename):
    image_out_path = os.path.join(out_dir, "test", "images", degrees, dirname, dirname_inner)
    hog_out_path = os.path.join(out_dir, "test", "hog", degrees, dirname, dirname_inner)
    if not os.path.exists(image_out_path):
        os.makedirs(image_out_path)
    if not os.path.exists(hog_out_path):
        os.makedirs(hog_out_path)
    np.save(os.path.join(hog_out_path, filename[:-4]), hog_fd)
    imsave(os.path.join(image_out_path, filename), image, format='JPEG')


def make_train(image, hog_fd, degrees, out_dir, dirname, dirname_inner, filename):
    image_out_path = os.path.join(out_dir, "train", "images", degrees, dirname, dirname_inner)
    hog_out_path = os.path.join(out_dir, "train", "hog", degrees, dirname, dirname_inner)
    if not os.path.exists(image_out_path):
        os.makedirs(image_out_path)
    if not os.path.exists(hog_out_path):
        os.makedirs(hog_out_path)
    np.save(os.path.join(hog_out_path,filename[:-4]), hog_fd)
    imsave(os.path.join(image_out_path, filename), image, format='JPEG')


# Rotates counter-clockwise
def rotate_image(imageArray, degrees):
    if degrees != 0:
        return rotate(imageArray, degrees)
    else:
        return imageArray


def generate_arrays(imageArray,visualise):
    image_to_hog = imageArray / 255.
    image_to_hog = rgb2gray(image_to_hog)
    if visualise:
        hog_features, hog_image = hog(image_to_hog, orientations=9, pixels_per_cell=(16, 16),
                                      cells_per_block=(1, 1), visualise=True)
        hog_fd = hog_features.astype(np.float32)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    else:
        hog_features = hog(image_to_hog, orientations=9, pixels_per_cell=(16, 16),
                           cells_per_block=(1, 1), visualise=False)
        hog_fd = hog_features.astype(np.float32)
        hog_image_rescaled = None
    return hog_fd, hog_image_rescaled


def save_image(imageArray, path):
    # np.save(os.path.join(path[:-4]), hog_features)
    imsave(os.path.join(path), imageArray, format='JPEG')


def load_image(image_file_path, PIL_image=False):
    if PIL_image:
        imageArray = Image.open(image_file_path,'r')
    else:
        imageArray = imread(image_file_path)
    return imageArray


def handler(image_dir, out_dir, save_array=False, visualise=False):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(os.path.join(out_dir, "train")):
        os.mkdir(os.path.join(out_dir, "train"))
    if not os.path.exists(os.path.join(out_dir, "train", "images")):
        os.mkdir(os.path.join(out_dir, "train", "images"))
    if not os.path.exists(os.path.join(out_dir, "train", "hog")):
        os.mkdir(os.path.join(out_dir, "train", "hog"))

    if not os.path.exists(os.path.join(out_dir, "test")):
        os.mkdir(os.path.join(out_dir, "test"))
    if not os.path.exists(os.path.join(out_dir, "test", "images")):
        os.mkdir(os.path.join(out_dir, "test", "images"))
    if not os.path.exists(os.path.join(out_dir, "test", "hog")):
        os.mkdir(os.path.join(out_dir, "test", "hog"))

    if not os.path.exists(os.path.join(out_dir, "valid")):
        os.makedirs(os.path.join(out_dir, "valid"))
    if not os.path.exists(os.path.join(out_dir, "valid", "images")):
        os.makedirs(os.path.join(out_dir, "valid", "images"))
    if not os.path.exists(os.path.join(out_dir, "valid", "hog")):
        os.makedirs(os.path.join(out_dir, "valid", "hog"))
    num_cur = 0
    count_total = 0
    t = time.time()
    num_imgs = 10000
    counter = {}
    for i in range(0, 4):
        counter[str(i*90)] = 0
    for root, dirnames, filenames in os.walk(image_dir):
        for dirname in dirnames:
            if save_array:
                img_train_dict = {}
                hog_train_dict = {}
                img_test_dict = {}
                hog_test_dict = {}
                img_valid_dict = {}
                hog_valid_dict = {}
            if dirname is not "P" and dirname is not "L" and dirname.count('Meta') == 0:
                print(dirname + ":")
                for root_inner, dirnames_inner, filenames_inner in os.walk(os.path.join(root, dirname)):
                    for dirname_inner in ["L", "P"]:
                        for root_actual, dirnames_extra, filenames_actual in \
                                os.walk(os.path.join(root_inner, dirname_inner)):
                            count_total = 0
                            count_valid = 0
                            count_test = 0
                            # Shuffle files to randomize as images next to each other are sometimes very similar
                            shuffle(filenames_actual)

                            for filename in filenames_actual:
                                if save_array and count_total % num_imgs == 0:
                                    for i in range(0, 4):
                                        img_train_dict[str(i * 90)] = []
                                        hog_train_dict[str(i * 90)] = []
                                        img_test_dict[str(i * 90)] = []
                                        hog_test_dict[str(i * 90)] = []
                                        img_valid_dict[str(i * 90)] = []
                                        hog_valid_dict[str(i * 90)] = []

                                image = load_image(os.path.join(root_actual,filename))
                                if count_total%20 == 0:
                                    degrees = (count_test % 4) * 90
                                    image = rotate_image(image, degrees)
                                    hog_fd, hog_image = generate_arrays(image, visualise)
                                    if save_array:
                                        img_test_dict[str(degrees)].append(image/255.)
                                        hog_test_dict[str(degrees)].append(hog_fd)
                                    else:
                                        make_test(image, hog_fd, str(degrees), out_dir, dirname, dirname_inner,
                                                  filename)
                                    count_test += 1
                                elif count_total % 20 == 1:
                                    degrees = (count_valid % 4) * 90
                                    image = rotate_image(image, degrees)
                                    hog_fd, hog_image = generate_arrays(image, visualise)
                                    if save_array:
                                        img_valid_dict[str(degrees)].append(image/255.)
                                        hog_valid_dict[str(degrees)].append(hog_fd)
                                    else:
                                        make_valid(image, hog_fd, str(degrees), out_dir, dirname, dirname_inner,
                                                   filename)
                                    count_valid += 1
                                else:
                                    degrees = ((count_total-count_test-count_valid) % 4) * 90
                                    counter[str(degrees)] += 1
                                    image = rotate_image(image, degrees)
                                    hog_fd, hog_image = generate_arrays(image, visualise)
                                    if save_array:
                                        img_train_dict[str(degrees)].append(image/255.)
                                        hog_train_dict[str(degrees)].append(hog_fd)
                                    else:
                                        make_train(image, hog_fd, str(degrees), out_dir, dirname, dirname_inner,
                                                   filename)
                                count_total += 1

                                if visualise:
                                    hog_meta_path = os.path.join(image_dir, dirname + "Meta", dirname_inner)
                                    if not os.path.exists(hog_meta_path):
                                        os.makedirs(hog_meta_path)

                                    imsave(os.path.join(hog_meta_path,filename[:-4] + ".hog"), hog_image, format='JPEG')

                                if save_array and count_total % num_imgs == 0:
                                    num_cur = str(
                                        int(math.ceil((count_total + (num_imgs - count_total) % num_imgs) / num_imgs)))
                                    print( dirname_inner + num_cur + " train: " + str(
                                        count_total - count_test - count_valid)
                                          + " test: " + str(count_test) + " valid: " + str(count_valid))
                                    save_pickle(out_dir, dirname + dirname_inner + num_cur, count_total, img_train_dict,
                                                hog_train_dict, img_test_dict, hog_test_dict, img_valid_dict,
                                                hog_valid_dict)

                        print(dirname_inner + " train: " + str(
                            count_total - count_test - count_valid)
                              + " test: " + str(count_test) + " valid: " + str(count_valid))

                        if save_array:
                            save_pickle(out_dir, dirname + dirname_inner + num_cur, count_total, img_train_dict,
                                        hog_train_dict, img_test_dict, hog_test_dict, img_valid_dict, hog_valid_dict)

                    break
        break
    print(time.time() - t)
    print(counter)
    print(count_total, count_test, count_valid)


def load_train(data_folder, tag, type, number, hog=False):
    if hog:
        if os.path.exists(os.path.join(data_folder, "train", "hog",tag+type+number+".pkl")):
            with open(os.path.join(data_folder, "train", "hog", tag + type + number + ".pkl"),'r') as f:
                return pickle.load(f)
        else:
            return None
    else:
        if os.path.exists(os.path.join(data_folder, "train", "images", tag + type + number + ".pkl")):
            with open(os.path.join(data_folder, "train", "images", tag + type + number + ".pkl"),'r') as f:
                return pickle.load(f)
        else:
            return None


def load_test(data_folder, tag, type, number, hog=False):
    if hog:
        if os.path.exists(os.path.join(data_folder, "test", "hog", tag + type + number + ".pkl")):
            with open(os.path.join(data_folder, "test", "hog", tag + type + number + ".pkl"), 'r') as f:
                return pickle.load(f)
        else:
            return None
    else:
        if os.path.exists(os.path.join(data_folder, "test", "images", tag + type + number + ".pkl")):
            with open(os.path.join(data_folder, "test", "images", tag + type + number + ".pkl"), 'r') as f:
                return pickle.load(f)
        else:
            return None


def load_valid(data_folder, tag, type, number, hog=False):
    if hog:
        if os.path.exists(os.path.join(data_folder, "valid", "hog", tag + type + number + ".pkl")):
            with open(os.path.join(data_folder, "valid", "hog", tag + type + number + ".pkl"), 'r') as f:
                return pickle.load(f)
        else:
            return None
    else:
        if os.path.exists(os.path.join(data_folder, "valid", "images", tag + type + number + ".pkl")):
            with open(os.path.join(data_folder, "valid", "images", tag + type + number + ".pkl"), 'r') as f:
                return pickle.load(f)
        else:
            return None


def resize_batch(input_folder,out_folder, resizeDims=(224,224)):
    os.mkdir(out_folder)
    for root, dirnames, filenames in os.walk(input_folder):
        for dirname in dirnames:
            if dirname is not "P" and dirname is not "L" and dirname.count('Meta') == 0:
                os.mkdir(os.path.join(out_folder,dirname))
                for root_inner, dirnames_inner, filenames_inner in os.walk(os.path.join(root, dirname)):
                    for dirname_inner in dirnames_inner:
                        os.mkdir(os.path.join(out_folder, dirname,dirname_inner))
                        for root_actual, dirnames_extra, filenames_actual \
                                in os.walk(os.path.join(root_inner, dirname_inner)):
                            for filename in filenames_actual:
                                imageArray = load_image(os.path.join(root_actual, filename), PIL_image=True)
                                imageArray = normalizeArray(imageArray, resizeDims=resizeDims)
                                imageArray.save(os.path.join(out_folder, dirname, dirname_inner, filename), 'JPEG')


# image_folder_depth: /home/sample/data/images/label1/label2/label3/image.jpg has a value of 3.
# ie. Number of folders after images folder
def hog_batch(input_folder, out_folder, image_folder_depth=3, label="hog"):
    corrupted = []
    for data_set in ["test", "train", "valid"]:
        image_list, label_list, tags_list = create_labeled_image_list(input_folder, data_set=data_set)
        count = 0
        for images in image_list:
            remaining = os.path.split(images)
            outpath = remaining[1]

            for i in range(image_folder_depth):
                remaining = os.path.split(remaining[0])
                outpath = os.path.join(remaining[1], outpath)
            outpath = os.path.join(out_folder, data_set, label, outpath)
            if not os.path.exists(os.path.split(outpath)[0]):
                os.makedirs(os.path.split(outpath)[0])
            img = load_image(images)
            hog_fd, hod_image = generate_arrays(img, False)
            pickle.dump(hog_fd, open(outpath[:-4] + '.npy', "wb"))

            test_str = open(outpath[:-4]+'.npy','rb').read()
            test = pickle.loads(test_str)
            if sum(hog_fd - test) > 0.00001:
                print("File " + outpath[:-4] + ".npy is corrupt")
                corrupted.append(outpath[:-4]+".npy")
            count += 1
        print(count)
    pickle.dump(corrupted, open(os.path.join(out_folder, "invalid_hog.log"), "wb"))


def split_SUN397(data_loc, out_folder, overwrite=False):
    data_loc = os.path.normpath(os.path.expandvars(os.path.expanduser(data_loc)))
    print(data_loc)
    test_set = []
    train_set = []
    valid_set = []
    count = 0
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    else:
        if not overwrite:
            print("File already exists")
            return
    for root, dirnames, filenames in os.walk(data_loc):
        # print(root)
        if len(dirnames) == 0:
            filenames = [os.path.join(root, x) for x in filenames]
            count += len(filenames)
            shuffle(filenames)
            test_set.extend(filenames[:len(filenames)//5])
            temp_set = filenames[len(filenames)//5:]
            valid_set.extend(temp_set[:len(temp_set)//5])
            train_set.extend(temp_set[len(temp_set)//5:])
    with open(os.path.join(out_folder, "valid.txt"), "w", newline='\n') as f:
        for filename in valid_set:
            f.write(filename + '\n')
    with open(os.path.join(out_folder, "test.txt"), "w", newline='\n') as f:
        for filename in test_set:
            f.write(filename + '\n')
    with open(os.path.join(out_folder, "train.txt"), "w", newline='\n') as f:
        for filename in train_set:
            f.write(filename + '\n')
    print(len(valid_set), len(test_set), len(train_set))
    print(count)


def test_create_labeled_image_list1(data_folder, data_set, feature):
    a, b, c = create_labeled_image_list(data_folder, data_set=data_set, feature=feature, num_images=None,
                                        labeled_data=True)
    for i in range(len(a)):
        print(a[i], b[i], c[i])
    print("Total: " + str(len(a)))


def test_create_labeled_image_list2(data_folder, data_set, feature):
    a, b, c = create_labeled_image_list(data_folder, data_set=data_set, feature=feature, num_images=None,
                                        labeled_data=False)
    for i in range(len(a)):
        print(a[i], b[i], c[i])
    print("Total: " + str(len(a)))


def test_open_single_file_with_tf(filename):
    with tf.device("/cpu:0"):
        sess = tf.Session()
        tf_file = tf.constant(filename, dtype=tf.string)
        tf_image = tf.read_file(tf_file)
        image = tf.image.decode_jpeg(tf_image, channels=3)
        image.eval(session=sess)


def test_each_file_with_tf(log_filename):
    # from contextlib import redirect_stderr, redirect_stdout

    with tf.device("/cpu:0"):
        sess = tf.Session()
        images, labels, tags = input_pipeline("D:\\PhotoOrientation", 1, data_set="SUN397", feature="images",
                                              orientations=[0], binary_file=False, num_epochs=1,
                                              num_images=None, labeled_data=False, rand_seed=None, num_threads=1)
        init_ops = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess.run(init_ops)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        steps = 0
        try:
            while not coord.should_stop():
                print(steps)
                imgs_list, labels_list, tags_list = sess.run([images, labels, tags])
                if len(tags_list) > 0:
                    print(tags_list[0].decode("UTF-8"))
                steps += 1
        except tf.errors.OutOfRangeError:
            print('Finished checking -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads=threads)


def test_read_file_format(data_loc):
    image_list = []
    data_loc = os.path.normpath(os.path.expandvars(os.path.expanduser(data_loc)))
    print(data_loc)
    print(os.listdir(data_loc))
    for filename in os.listdir(data_loc):
        if os.path.isfile(os.path.join(data_loc, filename)) and filename.count(".jpg") > 0:
            image_list.append(os.path.join(data_loc, filename))
    from scipy.misc import imsave, imshow, imrotate, imread, imresize
    import vgg16
    import numpy as np
    sess = tf.Session()
    batch_size = 1
    imgs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name="Inputs")
    y_ = tf.placeholder(tf.int32, shape=(batch_size), name="Outputs")
    learning_rate = .0001
    globalStep = tf.Variable(0, name='global_step', trainable=False)

    snapshot = "~/Downloads/5.pkl"
    snapshot = os.path.normpath(os.path.expandvars(os.path.expanduser(snapshot)))
    M = np.load(snapshot)
    vgg = vgg16.VGG16(imgs, y_, learning_rate, max_pool_num=5, global_step=globalStep, snapshot=M)

    init = tf.initialize_all_variables()
    sess.run(init)
    #image_file = tf.placeholder(tf.string)
    #tensor_image = tf.read_file(image_file)
    #image = tf.image.decode_jpeg(tensor_image, channels=3)
    #image = tf.image.resize_images(image, [224, 224])

    for images in image_list:
        # image = tf.image.rot90(image, k=1)
        # multiplier = tf.div(tf.constant(224, tf.float32),
        #                     tf.cast(tf.maximum(tf.shape(image)[0], tf.shape(image)[1]), tf.float32))
        # x = tf.cast(tf.round(tf.mul(tf.cast(tf.shape(image)[0], tf.float32), multiplier)), tf.int32)
        # y = tf.cast(tf.round(tf.mul(tf.cast(tf.shape(image)[1], tf.float32), multiplier)), tf.int32)
         #, method=tf.image.ResizeMethod.BICUBIC)
        #im = sess.run(image, feed_dict={image_file: images})
        print(images)
        im = imread(images)
        im = imresize(im, (224, 224))
        probs = sess.run(vgg.probs, feed_dict={vgg.inputs: [im], vgg.keep_probs: 1})[0]
        # image = tf.image.resize_images(image, [224, 224])
        #
        # print(x.eval(session=sess))
        # print(y.eval(session=sess))
        # image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
        print(probs)
        x = 90*np.argmax(probs)
        print(x)
        imsave(data_loc + "/rotation1/" + str(x) + "-" + os.path.split(images)[1], imrotate(im, int(x)), "JPEG")
        #
        # imsave(data_loc + "/resized.jpg", image, format='JPEG')
        # os.execl(data_loc, "open resized.jpg")

        # imshow(im)

    sess.close()


def test_create_labeled_image_list_from_file(data_folder):
    images, labels, tags = create_labeled_image_list_from_file(data_folder, "train", feature="images",
                                                               num_images=1000, labeled_data=False)
    print(len(images), len(labels), len(tags))
    print(images[0], labels[0], tags[0])


def test_sets_files(data_folder, feature):
    with open(os.path.join(data_folder, feature, "valid.txt")) as f:
        filenames = f.readlines()
        print(len(filenames))
    with open(os.path.join(data_folder, feature, "test.txt")) as f:
        filenames = f.readlines()
        print(len(filenames))
    with open(os.path.join(data_folder, feature, "train.txt")) as f:
        filenames = f.readlines()
        print(len(filenames))


def test_input_pipeline_from_file(directory):
    images_batch, labels_batch, tags_batch = input_pipeline(directory, 10, data_set="test", feature="images",
                                                            orientations=[0], num_epochs=1, num_images=None,
                                                            labeled_data=False, rand_seed=None, num_threads=10,
                                                            from_file=True)
    sess = tf.Session()
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    x = 0
    step = 0
    try:
        while not coord.should_stop():
            print(step)
            images, label, tags = sess.run([images_batch, labels_batch, tags_batch])
            x += len(tags)
            step += 1
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()
    coord.join(threads)
    print(x)


def run_tests():
    data_loc = "C:\\PhotoOrientation\\SUN397\\images"
    outfolder = "C:\\PhotoOrientation\\SUN397\\sets1"
    # for root, dirs, files in os.walk("D:\\PhotoOrientation\\SUN397\\fixes\\converted_images1"):
    #     for filename in files:
    #         test_open_single_file_with_tf(filename=os.path.join(root,filename))
    # test_create_labeled_image_list1("D:\\PhotoOrientation", "sun397", "images")
    # test_create_labeled_image_list1("D:\\PhotoOrientation", "sun397", "images1")

    # with open("C:\\PhotoOrientation\\data\\SUN397\\Logs\\incorrect_endings.txt", 'r') as f:
    #    for line in f:
    #        filename = line.replace('\n', '')
    #        test_open_single_file_with_tf(filename)

    # test_each_file_with_tf("D:\\PhotoOrientation\\SUN397\\err_log.txt")
    # test_create_labeled_image_list_from_file("C:\\PhotoOrientation\\SUN397\\sets1")
    # split_SUN397(data_loc, outfolder + "\\images", overwrite=True)
    # test_input_pipeline_from_file(outfolder)
    # test_sets_files(outfolder, "images")


def run_tests_mac():
    # split_SUN397("~/Documents/SUN397/images", "")
    test_read_file_format("~/Documents")

if __name__ == "__main__":
    t = int(time.time())
    # t = 1454219613
    print("t=", t)
    # hog_batch("/home/ujash/nvme/data2","/home/ujash/nvme/data2", label="hog2")
    # resize_batch("/home/ujash/images_flickr/down1","/home/ujash/images_flickr/down4")
    # handler("/home/ujash/nvme/down4","/home/ujash/nvme/data2")
    run_tests()
    # run_tests_mac()
