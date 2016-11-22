from shutil import copyfile
import os
import numpy as np

def copy_incorrect(in_folder, out_folder):
    f = open(os.path.join(in_folder, "stats", "snapshotVGG1-5-test.txt"), "r")
    page = f.read()

    sources = page.split('\n')
    print(len(sources))
    count = 0
    for source in sources:
        if source.find("jpg") >= 0:
            count+=1
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

if __name__ == "__main__":
    data_folder_loc = os.path.join("C:\\PhotoOrientation", "SUNdatabase", "images")
    # copy_incorrect(data_folder, data_folder)
    convert_files_to_jpeg(data_folder_loc)

