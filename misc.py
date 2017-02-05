from shutil import copyfile
import os
import csv
from scipy.misc import imread, imsave
import numpy as np
from PIL import Image
import tensorflow as tf
from datahandler import input_pipeline


def find_num_images_by_tag(data_folder, class_names_file=None):
    print(class_names_file)
    num_images_by_tag = {}
    if class_names_file:
        with open(class_names_file, "r") as class_names:
            class_data_folders = class_names.read().split('\n')
            class_data_folders.remove("")
            for i in range(len(class_data_folders)):
                if class_data_folders[i].count("/") > 0:
                    class_data_folders[i] = os.path.join(*class_data_folders[i].split("/"))
                elif class_data_folders[i].count("\\") > 0:
                    class_data_folders[i] = os.path.join(*class_data_folders[i].split("\\"))
            print(class_data_folders)
            print(len(class_data_folders))
    else:
        class_data_folders = [""]

    for class_data_folder in class_data_folders:
        for root, dirnames, filenames in os.walk(os.path.join(data_folder, class_data_folder)):
            if class_data_folder == "":
                tag = os.path.split(root)[1]
            else:
                tag = class_data_folder
            for filename in filenames:
                if tag not in num_images_by_tag:
                    num_images_by_tag[tag] = filename.count(".jpg")
                else:
                    num_images_by_tag[tag] += filename.count(".jpg")
    return num_images_by_tag


def copy_incorrect(in_folder, out_folder, incorrect_files="snapshotVGG1-5-test.txt"):
    from scipy.misc import imread, imsave, imrotate
    print(incorrect_files)
    if os.path.exists(incorrect_files):
        f = open(incorrect_files, "r")
        print("File found")
    else:
        f = open(os.path.join(in_folder, "stats", incorrect_files), "r")
    page = f.read()

    sources = page.split('\n')
    print(sources)
    print(len(sources))
    count = 0
    for source in sources:
        if source.find("jpg") >= 0:
            fileinfo = source
            if source.find(",") >= 0:
                fileinfo = source.split(", ")[0]
                rotation = source.split(", ")[1]
                image = imread(fileinfo)
                image = imrotate(image, int(rotation))
            else:
                image = imread(fileinfo)
            if count == 0:
                print(fileinfo)
            count += 1
            destination = os.path.split(fileinfo.replace(in_folder, out_folder))[0]
            if not os.path.exists(destination):
                os.makedirs(destination)
            filename = os.path.split(fileinfo)[1]
            # print(os.path.join(destination, filename))
            imsave(os.path.join(destination, filename), image)
    print("Moved " + str(count) + " files")


# Created to fix files in the SUN397 database. Some files are not truly JPEG, just have the extension changed.
# Tensorflow can only open JPEG/PNG images by default and checking for those differences with incorrect extensions
# is difficult.
# 0xff 0xd8 = JPEG and Ends with 0xff 0xd9
# 0x42 0x4d = BMP
# 0x47 0x49 = GIF
# 0x89 0x50 = PNG
def convert_files_to_jpeg(data_folder, outfolder):
    import binascii
    from scipy.misc import imread, imsave
    count = 0
    wrong_file = 0
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    inc_files = open(os.path.join(outfolder, "not_jpg_files.txt"), "w")
    # Find files with incorrect starting
    for root_inner, dir_inner, files in os.walk(data_folder):
        for file_name in files:
            orig_file = os.path.join(root_inner, file_name)
            with open(orig_file, 'rb') as f:
                page = f.read(2)
            page = binascii.hexlify(page)
            count += 1
            # print(page)
            if page != b'ffd8':
                print(page.decode("UTF-8") + ": " + orig_file)
                if orig_file.find('.db') >= 0:
                    os.remove(orig_file)
                    continue
                image = imread(orig_file)
                temp_fixed_dir = root_inner.replace(data_folder, os.path.join(outfolder, "converted_images1"))
                if not os.path.exists(temp_fixed_dir):
                    os.makedirs(temp_fixed_dir)
                temp_file_name = os.path.join(temp_fixed_dir, file_name)
                imsave(os.path.join(temp_file_name), image, format='JPEG')

                temp_incorrect_dir = root_inner.replace(data_folder, os.path.join(outfolder, "saved_orig_images1"))
                if not os.path.exists(temp_incorrect_dir):
                    os.makedirs(temp_incorrect_dir)
                temp_file_name = os.path.join(temp_incorrect_dir, file_name)
                copyfile(orig_file, temp_file_name)

                wrong_file += 1
                inc_files.write(orig_file + "," + page.decode("UTF-8") + "\n")
    inc_files.close()
    '''
    # Find files with bad ending
    # This approach does not seem to work as file might not end with 0xffd9 and tensorflow has no problems opening them.
    # Best would be to use tensorflow itself to find out which files it has problems opening.
    for root_inner, dir_inner, files in os.walk(data_folder):
        for file_name in files:
            orig_file = os.path.join(root_inner, file_name)
            with open(orig_file, 'rb') as f:
                page = f.read()
            page = binascii.hexlify(page)
            count += 1
            if page[-4:] != b'ffd9':
                wrong_file += 1
                print(page[-10:])
                print(orig_file)

                image = imread(orig_file)

                temp_fixed_dir = root_inner.replace("images", "fixed_images1")
                if not os.path.exists(temp_fixed_dir):
                    os.makedirs(temp_fixed_dir)
                temp_file = os.path.join(temp_fixed_dir, file_name)
                imsave(os.path.join(temp_file), image, format='JPEG')

                temp_incorrect_dir = root_inner.replace("images", "saved_incorrect_images1")
                if not os.path.exists(temp_incorrect_dir):
                    os.makedirs(temp_incorrect_dir)
                temp_file_name = os.path.join(temp_incorrect_dir, file_name)
                copyfile(orig_file, temp_file_name)
                '''
    print("Total files: " + str(count))
    print("Incorrect files: " + str(wrong_file))


def write_dict_to_csv(data_set_stats, data_stats_folder, stat_filename, col_keys=None):
    if not col_keys:
        col_keys = ["Value"]
    if not os.path.exists(data_stats_folder):
        os.makedirs(data_stats_folder)
    with open(os.path.join(data_stats_folder, stat_filename + ".csv"), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, lineterminator='\n')
        writer.writerow(["Tag"] + col_keys)
        keys = sorted(data_set_stats.keys())
        for key in keys:
            row_list = [key]
            if len(col_keys) > 1:
                for col_key in col_keys:
                    row_list.append(data_set_stats[key][col_key])
            else:
                row_list.append(data_set_stats[key])
            writer.writerow(row_list)


def find_corrupt_in_log(logfile):
    print_next = False
    count = 0
    with open(logfile, "r") as f:
        for line in f:
            if print_next and line.find("jpg") > 0:
                count += 1
                print(os.path.join("D:\\PhotoOrientation\\SUN397\\images\\", line),end='')
                print_next = False
            if line.find("Corrupt") >= 0:
                print_next = True


def inputs():
    filenames = ['img1.jpg', 'img2.jpg' ]
    filename_queue = tf.train.string_input_producer(filenames,num_epochs=1)
    read_input = read_image(filename_queue)
    reshaped_image = modify_image(read_input)
    return reshaped_image


def blend_images(data_folder1, data_folder2, out_folder, alpha=.5):
    filename_queue = tf.placeholder(dtype=tf.string)
    label = tf.placeholder(dtype=tf.int32)
    tensor_image = tf.read_file(filename_queue)

    image = tf.image.decode_jpeg(tensor_image, channels=3)

    multiplier = tf.div(tf.constant(224, tf.float32),
                        tf.cast(tf.maximum(tf.shape(image)[0], tf.shape(image)[1]), tf.float32))
    x = tf.cast(tf.round(tf.mul(tf.cast(tf.shape(image)[0], tf.float32), multiplier)), tf.int32)
    y = tf.cast(tf.round(tf.mul(tf.cast(tf.shape(image)[1], tf.float32), multiplier)), tf.int32)
    image = tf.image.resize_images(image, [x, y])

    image = tf.image.rot90(image, k=label)

    image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
    sess = tf.Session()
    sess.run(tf.local_variables_initializer())
    for root, folders, files in os.walk(data_folder1):
        for each in files:
            if each.find('.jpg') >= 0:
                img1 = Image.open(os.path.join(root, each))
                img2_path = os.path.join(root.replace(data_folder1, data_folder2), each.split("-")[-1])
                rotation = int(each.split("-")[1])
                img2 = sess.run(image, feed_dict={filename_queue: img2_path, label: rotation})
                imsave(os.path.join(os.getcwd(), "temp", "temp.jpg"), img2)
                img2 = Image.open(os.path.join(os.getcwd(), "temp", "temp.jpg"))
                out_image = Image.blend(img1, img2, alpha)
                outfile = os.path.join(root.replace(data_folder1, out_folder), each)
                if not os.path.exists(os.path.split(outfile)[0]):
                    os.makedirs(os.path.split(outfile)[0])
                out_image.save(outfile)
            else:
                print(each)
    sess.close()


def add_value_to_images(data_folder, out_folder, value):
    for root, folders, files in os.walk(data_folder):
        for each in files:
            if each.find('.jpg') >= 0:
                images = imread(os.path.join(root, each))
                print(images)
                images = images + value
                print(images)
                break
                outfile = os.path.join(root.replace(data_folder, out_folder), each)
                if not os.path.exists(os.path.split(outfile)[0]):
                    os.makedirs(os.path.split(outfile)[0])
                imsave(outfile, images)
            else:
                print(each)


def rename_grad_desc_files():
    for root, folder, filenames in os.walk(os.path.join(os.getcwd(), "temp", "grad_desc_neg-pos_0")):
        print(root)
        for each in filenames:

            if each.count(".jpg") > 0:
                if each.count("-prob") > 0:

                    replacement_name = each.split("-prob")

                    replacement_name[-1] = (replacement_name[-1].split("sun_")[-1]).split(".")[0]

                    new_filename = "sun_" + replacement_name[-1] + "-" + replacement_name[0] + ".jpg"

                    new_filepath = os.path.join(root.replace("grad_desc_neg-pos_0", "gd-0"), new_filename)

                    if not os.path.exists(os.path.split(new_filepath)[0]):
                        os.makedirs(os.path.split(new_filepath)[0])
                    os.rename(os.path.join(root, each), new_filepath)


def rename_grad_desc_files2():
    for root, folder, filenames in os.walk(os.path.join(os.getcwd(), "temp", "gd-0")):
        print(root)
        for each in filenames:

            if each.count(".jpg") > 0:
                if each.startswith("prob"):

                    replacement_name = each.split("-sun_")

                    replacement_name[-1] = (replacement_name[-1].split("sun_")[-1]).split(".")[0]

                    new_filename = "sun_" + replacement_name[-1] + "-" + replacement_name[0] + ".jpg"

                    new_filepath = os.path.join(root, new_filename)

                    if not os.path.exists(os.path.split(new_filepath)[0]):
                        os.makedirs(os.path.split(new_filepath)[0])
                    os.rename(os.path.join(root, each), new_filepath)


if __name__ == "__main__":
    rename_grad_desc_files2()
    '''
    data_folder_loc = os.path.join("C:", os.sep, "PhotoOrientation", "SUN397", "images")
    for imgs in ["incorrect", "correct"]:
        outfolder_loc = os.path.join(os.getcwd(), "temp", "gradient_desc5", imgs, "images")
        infolder_loc = os.path.join(os.getcwd(), "temp", "gradient_desc2", imgs, "images")
        blend_images(infolder_loc, data_folder_loc, outfolder_loc, .1)
    '''
    '''
    outfolder_loc = os.path.join(os.getcwd(), "temp", "CorelDB", "nonJPEG1")
    print(outfolder_loc)
    data_loc = "C:\\PhotoOrientation\\CorelDB\\images"

    convert_files_to_jpeg(data_loc, outfolder_loc)
    '''
    '''
    inc_file = "temp\\incorrect.txt"
    copy_incorrect(data_loc, outfolder_loc,incorrect_files=inc_file)
    '''

    '''data_folder_loc = os.path.join("D:\\PhotoOrientation", "SUN397", "incorrect")
    outfolder = os.path.join("D:", os.sep, "PhotoOrientation", "SUN397", "fixes")


    find_corrupt_in_log("C:\PhotoOrientation\data\SUN397\Logs\log_errors.txt")
    copy_incorrect("D:\\PhotoOrientation\\SUN397", "D:\\PhotoOrientation\\SUN397",
                   "C:\\PhotoOrientation\\data\\SUN397\\Logs\\incorrect_endings.txt")
    '''

    '''
    image_nums = find_num_images_by_tag(data_folder_loc,
                                        os.path.join(os.path.split(data_folder_loc)[0], "ClassName.txt"))
    for tag in image_nums:
        print(tag + ": " + str(image_nums[tag]))
    write_dict_to_csv(image_nums, os.path.join(os.path.split(data_folder_loc)[0], "stats"), "data_info",
                      col_keys=["Num Images"])

    # copy_incorrect(data_folder, data_folder)
    # convert_files_to_jpeg(data_folder_loc, outfolder)

        # print("err")
    # print(count)

    cur_dir = os.getcwd()
    print("resizing images")
    print("current directory:",cur_dir)

    with tf.Graph().as_default():
        image_batch, label_batch, tag_batch = input_pipeline(data_folder_loc, batch_size, data_set="train",
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
        image = inputs()
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        for i in xrange(2):
            img = sess.run(image)
            img = Image.fromarray(img, "RGB")
    '''
