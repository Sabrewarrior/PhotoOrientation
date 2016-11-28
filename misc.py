from shutil import copyfile
import os
import csv
import numpy as np


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


def copy_incorrect(in_folder, out_folder):
    f = open(os.path.join(in_folder, "stats", "snapshotVGG1-5-test.txt"), "r")
    page = f.read()

    sources = page.split('\n')
    print(len(sources))
    count = 0
    for source in sources:
        if source.find("jpg") >= 0:
            count += 1
            res, filename = os.path.split(source)
            res, layout = os.path.split(res)
            res, tag = os.path.split(res)
            res, orientation = os.path.split(res)
            destination = os.path.join(out_folder, "incorrect", orientation, tag, layout)
            if not os.path.exists(destination):
                os.makedirs(destination)
            copyfile(source, os.path.join(destination, filename))

    print("Moved " + str(count) + " files")


# Created to fix files in the SUN397 database. Some files are not truly JPEG, just have the extension changed.
# Tensorflow can only open JPEG/PNG images by default and checking for those differences with incorrect extensions
# is difficult.
# 0xff 0xd8 = JPEG and Ends with 0xff 0xd9
# 0x42 0x4d = BMP
# 0x47 0x49 = GIF
# 0x89 0x50 = PNG
def convert_files_to_jpeg(data_folder):
    import binascii
    from scipy.misc import imread, imsave
    count = 0
    wrong_file = 0


    # Find files with incorrect starting
    for root_inner, dir_inner, files in os.walk(data_folder):
        for file_name in files:
            orig_file = os.path.join(root_inner, file_name)
            with open(orig_file, 'rb') as f:
                page = f.read(2)
            page = binascii.hexlify(page)
            count += 1
            print(page)
            if page != b'ffd8':
                # image = imread(org_file)

                # temp_fixed_dir = root_inner.replace("images", "fixed_images")
                # if not os.path.exists(temp_fixed_dir):
                #     os.makedirs(temp_fixed_dir)
                # temp_file = os.path.join(temp_fixed_dir, file_name)
                # imsave(os.path.join(temp_file), image, format='JPEG')

                temp_incorrect_dir = root_inner.replace("images", "saved_incorrect_images")
                if not os.path.exists(temp_incorrect_dir):
                    os.makedirs(temp_incorrect_dir)
                temp_file_name = os.path.join(temp_incorrect_dir, file_name)
                copyfile(orig_file, temp_file_name)

                wrong_file += 1
                print(orig_file)

    '''
    # Find files with bad ending
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


if __name__ == "__main__":
    data_folder_loc = os.path.join("D:\\PhotoOrientation", "SUN397", "images")
    image_nums = find_num_images_by_tag(data_folder_loc,
                                        os.path.join(os.path.split(data_folder_loc)[0], "ClassName.txt"))
    for tag in image_nums:
        print(tag + ": " + str(image_nums[tag]))
    write_dict_to_csv(image_nums, os.path.join(os.path.split(data_folder_loc)[0], "stats"), "data_info",
                      col_keys=["Num Images"])

    # copy_incorrect(data_folder, data_folder)
    # convert_files_to_jpeg(data_folder_loc)

