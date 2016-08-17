import os
import math
import time
from random import shuffle, seed
from scipy.misc import imread, imsave
from pylab import float32
from skimage.color import rgb2gray
import numpy as np
from skimage.feature import hog
from skimage import exposure
import cPickle
from skimage.transform import rotate
from PIL import Image

def save_pickle(out_dir, tag_name, count_total, img_train_dict, hog_train_dict, img_test_dict, hog_test_dict,
                img_valid_dict, hog_valid_dict):
    for key in img_train_dict.keys():
        img_train_dict[key] = np.asarray(img_train_dict[key], dtype=float32)
        hog_train_dict[key] = np.asarray(hog_train_dict[key], dtype=float32)
        img_test_dict[key] = np.asarray(img_test_dict[key], dtype=float32)
        hog_test_dict[key] = np.asarray(hog_test_dict[key], dtype=float32)
        img_valid_dict[key] = np.asarray(img_valid_dict[key], dtype=float32)
        hog_valid_dict[key] = np.asarray(hog_valid_dict[key], dtype=float32)

    print "Saving " + tag_name
    with open(os.path.join(out_dir,"train", "images", tag_name+".pkl"), "wb") as img_train_file:
        cPickle.dump(img_train_dict, img_train_file, protocol=2)
    with open(os.path.join(out_dir, "train", "hog", tag_name + ".pkl"), "wb") as hog_train_file:
        cPickle.dump(hog_train_dict, hog_train_file, protocol=2)
    with open(os.path.join(out_dir, "test", "images", tag_name + ".pkl"), "wb") as img_test_file:
        cPickle.dump(img_test_dict, img_test_file, protocol=2)
    with open(os.path.join(out_dir, "test", "hog", tag_name + ".pkl"), "wb") as hog_test_file:
        cPickle.dump(hog_test_dict, hog_test_file, protocol=2)
    with open(os.path.join(out_dir, "valid", "images", tag_name + ".pkl"), "wb") as img_valid_file:
        cPickle.dump(img_valid_dict, img_valid_file, protocol=2)
    with open(os.path.join(out_dir, "valid", "hog", tag_name + ".pkl"), "wb") as hog_valid_file:
        cPickle.dump(hog_valid_dict, hog_valid_file, protocol=2)


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
        hog_features = hog_features.astype(float32)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    else:
        hog_features = hog(image_to_hog, orientations=9, pixels_per_cell=(16, 16),
                                      cells_per_block=(1, 1), visualise=False)
        hog_image_rescaled = None
    return hog_features, hog_image_rescaled

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
                print dirname + ":"
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
                                    print dirname_inner + num_cur + " train: " + str(
                                        count_total - count_test - count_valid) \
                                          + " test: " + str(count_test) + " valid: " + str(count_valid)
                                    save_pickle(out_dir, dirname + dirname_inner + num_cur, count_total, img_train_dict,
                                                hog_train_dict, img_test_dict, hog_test_dict, img_valid_dict,
                                                hog_valid_dict)

                        print dirname_inner + " train: " + str(
                            count_total - count_test - count_valid) \
                              + " test: " + str(count_test) + " valid: " + str(count_valid)

                        if save_array:
                            save_pickle(out_dir, dirname + dirname_inner + num_cur, count_total, img_train_dict,
                                    hog_train_dict, img_test_dict, hog_test_dict, img_valid_dict,
                                    hog_valid_dict)

                    break
        break
    print time.time() - t
    print counter
    print count_total, count_test, count_valid

def load_train(data_folder, tag, type, number, hog=False):
    if hog:
        if os.path.exists(os.path.join(data_folder, "train", "hog",tag+type+number+".pkl")):
            with open(os.path.join(data_folder, "train", "hog", tag + type + number + ".pkl"),'r') as f:
                return cPickle.load(f)
        else:
            return None
    else:
        if os.path.exists(os.path.join(data_folder, "train", "images", tag + type + number + ".pkl")):
            with open(os.path.join(data_folder, "train", "images", tag + type + number + ".pkl"),'r') as f:
                return cPickle.load(f)
        else:
            return None

def load_test(data_folder, tag, type, number, hog=False):
    if hog:
        if os.path.exists(os.path.join(data_folder, "test", "hog", tag + type + number + ".pkl")):
            with open(os.path.join(data_folder, "test", "hog", tag + type + number + ".pkl"), 'r') as f:
                return cPickle.load(f)
        else:
            return None
    else:
        if os.path.exists(os.path.join(data_folder, "test", "images", tag + type + number + ".pkl")):
            with open(os.path.join(data_folder, "test", "images", tag + type + number + ".pkl"), 'r') as f:
                return cPickle.load(f)
        else:
            return None

def load_valid(data_folder, tag, type, number, hog=False):
    if hog:
        if os.path.exists(os.path.join(data_folder, "valid", "hog", tag + type + number + ".pkl")):
            with open(os.path.join(data_folder, "valid", "hog", tag + type + number + ".pkl"), 'r') as f:
                return cPickle.load(f)
        else:
            return None
    else:
        if os.path.exists(os.path.join(data_folder, "valid", "images", tag + type + number + ".pkl")):
            with open(os.path.join(data_folder, "valid", "images", tag + type + number + ".pkl"), 'r') as f:
                return cPickle.load(f)
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


if __name__ == "__main__":
    t = int(time.time())
    # t = 1454219613
    print "t=", t
    seed(t)

    #resize_batch("/home/ujash/images_flickr/down1","/home/ujash/images_flickr/down4")

    handler("/home/ujash/nvme/down4","/home/ujash/nvme/data2")
