import os
import math
import time
from random import shuffle, seed
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


def read_file_format(filename_queue, binary=False):
    label = tf.cast(filename_queue[1], tf.int32)
    if binary:
        image = tf.read_file(filename_queue[0])
    else:
        tensor_image = tf.read_file(filename_queue[0])
        image = tf.image.decode_jpeg(tensor_image, channels = 3)
    return image, label


def input_pipeline(directory, batch_size, data_set="train", feature="images", num_epochs=None, num_images=None):
    if feature.count("images") > 0:
        binary = False
    else:
        binary = True
    with tf.name_scope('InputPipeline'):
        # Reads paths of images together with their labels
        image_list, label_list = create_labeled_image_list(directory, data_set=data_set, feature=feature,
                                                           num_images=num_images)

        # Makes an input queue
        input_queue = tf.train.slice_input_producer([image_list, label_list],num_epochs=num_epochs)
        image, label = read_file_format(input_queue, binary=binary)

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
                                                          min_after_dequeue=min_after_dequeue, num_threads=10)
        else:
            image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, capacity=capacity,
                                  num_threads=10)
        return image_batch, label_batch


def convert_binary_to_array(image_binary_list):
    images = []
    #print(image_binary_list)
    for image_binary in image_binary_list:
        images.append(pickle.loads(image_binary))
    return images


def convert_to_array(image_list):
    images = []
    #print(image_binary_list)
    for image in image_list:
        images.append(imread(image))
    return images


def create_labeled_image_list(directory, data_set="train", feature="images", num_images=None):
    image_list = []
    label_list = []
    directory = os.path.join(directory, data_set, feature)
    limit = 0
    if not os.path.exists(directory):
        print("Feature or dataset does not exist")
    for root, dirnames, filenames in os.walk(directory):
        print("Loading images from: " + root)
        for dirname in dirnames:
            if dirname == '0':
                label = 0
            elif dirname == '90':
                label = 1
            elif dirname == '180':
                label = 2
            else:
                label = 3
            for root_inner, dirnames_inner, filenames_actual in os.walk(os.path.join(root, dirname)):
                for filename in filenames_actual:
                    if num_images is not None:
                        if limit == num_images:
                            break
                    image_list.append(os.path.join(root_inner,filename))
                    label_list.append(label)
                    limit+=1
        break

    print(str(len(image_list)) + " images loaded")
    return image_list, label_list


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
    np.save(os.path.join(hog_out_path,filename[:-4]),hog_fd)
    imsave(os.path.join(image_out_path,filename),image,format='JPEG')


# Rotates counter-clockwise
def rotate_image(imageArray, degrees):
    if degrees !=0:
        return rotate(imageArray, degrees)
    else:
        return imageArray


def generate_arrays(imageArray,visualise):
    image_to_hog = imageArray / 255.
    image_to_hog = rgb2gray(image_to_hog)
    if visualise:
        hog_features, hog_image = hog(image_to_hog, orientations=9, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualise = True)
        hog_fd = hog_features.astype(np.float32)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    else:
        hog_features = hog(image_to_hog, orientations=9, pixels_per_cell=(16, 16),
                                      cells_per_block=(1, 1), visualise=False)
        hog_fd = hog_features.astype(np.float32)
        hog_image_rescaled = None
    return hog_fd, hog_image_rescaled


def save_image(imageArray, path):
    #np.save(os.path.join(path[:-4]), hog_features)
    imsave(os.path.join(path), imageArray, format='JPEG')


def load_image(image_file_path, PIL_image=False):
    if PIL_image:
        imageArray = Image.open(image_file_path,'r')
    else:
        imageArray = imread(image_file_path)
    return imageArray


def handler(image_dir, out_dir, save_array=False, visualise = False):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(os.path.join(out_dir,"train")):
        os.mkdir(os.path.join(out_dir, "train"))
    if not os.path.exists(os.path.join(out_dir,"train","images")):
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
    if not os.path.exists(os.path.join(out_dir, "valid","images")):
        os.makedirs(os.path.join(out_dir, "valid", "images"))
    if not os.path.exists(os.path.join(out_dir, "valid","hog")):
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
                for root_inner, dirnames_inner, filenames_inner in os.walk(os.path.join(root,dirname)):
                    for dirname_inner in ["L","P"]:
                        for root_actual, dirnames_extra, filenames_actual in os.walk(os.path.join(root_inner,dirname_inner)):
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
                                        make_test(image,hog_fd,str(degrees),out_dir,dirname,dirname_inner,filename)
                                    count_test+=1
                                elif count_total%20 == 1:
                                    degrees = (count_valid % 4) * 90
                                    image = rotate_image(image, degrees)
                                    hog_fd, hog_image = generate_arrays(image, visualise)
                                    if save_array:
                                        img_valid_dict[str(degrees)].append(image/255.)
                                        hog_valid_dict[str(degrees)].append(hog_fd)
                                    else:
                                        make_valid(image,hog_fd,str(degrees),out_dir,dirname,dirname_inner,filename)
                                    count_valid += 1
                                else:
                                    degrees = ((count_total-count_test-count_valid) % 4) * 90
                                    counter[str(degrees)]+=1
                                    image = rotate_image(image, degrees)
                                    hog_fd, hog_image = generate_arrays(image, visualise)
                                    if save_array:
                                        img_train_dict[str(degrees)].append(image/255.)
                                        hog_train_dict[str(degrees)].append(hog_fd)
                                    else:
                                        make_train(image,hog_fd,str(degrees),out_dir,dirname,dirname_inner,filename)
                                count_total+=1

                                if visualise:
                                    hog_meta_path = os.path.join(image_dir, dirname + "Meta", dirname_inner)
                                    if not os.path.exists(hog_meta_path):
                                        os.makedirs(hog_meta_path)

                                    imsave(os.path.join(hog_meta_path,filename[:-4] + ".hog"), hog_image, format='JPEG')

                                if save_array and count_total % num_imgs == 0:
                                    num_cur = str(
                                        int(math.ceil((count_total + (num_imgs - count_total) % num_imgs) / num_imgs)))
                                    print( dirname_inner + num_cur + " train: " + str(
                                        count_total - count_test - count_valid) \
                                          + " test: " + str(count_test) + " valid: " + str(count_valid))
                                    save_pickle(out_dir, dirname + dirname_inner + num_cur, count_total, img_train_dict,
                                                hog_train_dict, img_test_dict, hog_test_dict, img_valid_dict,
                                                hog_valid_dict)

                        print(dirname_inner + " train: " + str(
                            count_total - count_test - count_valid) \
                              + " test: " + str(count_test) + " valid: " + str(count_valid))

                        if save_array:
                            save_pickle(out_dir, dirname + dirname_inner + num_cur, count_total, img_train_dict,
                                    hog_train_dict, img_test_dict, hog_test_dict, img_valid_dict,
                                    hog_valid_dict)

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
                                imageArray = normalizeArray(imageArray, resizeDims = resizeDims)
                                imageArray.save(os.path.join(out_folder,dirname,dirname_inner,filename),'JPEG')


# image_folder_depth: /home/sample/data/images/label1/label2/label3/image.jpg has a value of 3.
# ie. Number of folders after images folder
def hog_batch(input_folder, out_folder, image_folder_depth=3,label="hog"):
    corrupted = []
    for data_set in ["test","train","valid"]:
        image_list, label_list = create_labeled_image_list(input_folder,data_set=data_set)
        count = 0
        for images in image_list:
            remaining = os.path.split(images)
            outpath = remaining[1]

            for i in range(image_folder_depth):
                remaining = os.path.split(remaining[0])
                outpath = os.path.join(remaining[1],outpath)
            outpath = os.path.join(out_folder,data_set,label,outpath)
            if not os.path.exists(os.path.split(outpath)[0]):
                os.makedirs(os.path.split(outpath)[0])
            img = load_image(images)
            hog_fd, hod_image = generate_arrays(img, False)
            pickle.dump(hog_fd,open(outpath[:-4]+'.npy',"wb"))

            test_str = open(outpath[:-4]+'.npy','rb').read()
            test = pickle.loads(test_str)
            if sum(hog_fd - test) > 0.00001:
                print("File " + outpath[:-4]+ ".npy is corrupt")
                corrupted.append(outpath[:-4]+".npy")
            count+= 1
        print(count)
    pickle.dump(corrupted, open(os.path.join(out_folder,"invalid_hog.log"),"wb"))

if __name__ == "__main__":
    t = int(time.time())
    # t = 1454219613
    print("t=", t)
    seed(t)
    #hog_batch("/home/ujash/nvme/data2","/home/ujash/nvme/data2", label="hog2")
    #resize_batch("/home/ujash/images_flickr/down1","/home/ujash/images_flickr/down4")

    #handler("/home/ujash/nvme/down4","/home/ujash/nvme/data2")
