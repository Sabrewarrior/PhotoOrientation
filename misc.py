from shutil import copyfile
import os


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


data_folder = os.path.join("C:\\PhotoOrientation", "data2")
copy_incorrect(data_folder, data_folder)
