import argparse
import pdb
import os
import zipfile
import csv


def get_parser(parser):
    parser.add_argument('--data_root', type = str, default = "E:\\code\\dataset\\blindness\\aptos2019-blindness-detection.zip")

    return parser.parse_args()


def transform_list(root, csv_file, name):
    file_path = os.path.join(root, '..', name)
    file = open(file_path, 'w')
    with open(csv_file) as f:
        f_csv = csv.reader(f)
        read_index = 0
        for row in f_csv:
            if read_index == 0:
                pass
            else:
                if len(row) == 2:
                    image_path = os.path.join(root, row[0])
                    label = os.path.join(row[1])
                    file.writelines(image_path + ' ' + str(label) + '\n')
                elif len(row) == 1:
                    image_path = os.path.join(root, row[0])
                    file.writelines(image_path + '\n')

            read_index += 1


def main():
    print("Start Processing!")
    parser = argparse.ArgumentParser(description="add blindness dataset path")
    flags = get_parser(parser)
    datapath = flags.data_root

    if datapath is None:
        raise  ValueError("data_root must be defined!")

    if not os.path.exists(datapath):
        raise ValueError("data_root {:} is not exists".format(dataPath))

    unzip_root = os.path.join(datapath, '..', "raw_data")
    if not os.path.exists(unzip_root):
        z = zipfile.ZipFile(datapath, 'r')
        z.extractall(path=unzip_root)
        z.close()

    print("Unzip data file done!")

    train_image_zip_path = os.path.join(unzip_root, "train_images.zip")
    train_image_unzip_path = os.path.join(unzip_root, "train_images")
    test_image_zip_path = os.path.join(unzip_root, "test_images.zip")
    test_image_unzip_path = os.path.join(unzip_root, "test_images")
    if not os.path.exists(train_image_unzip_path):
        z = zipfile.ZipFile(train_image_zip_path, 'r')
        z.extractall(path=train_image_unzip_path)
        z.close()
        os.remove(train_image_zip_path)
    print("Unzip train images file done!")

    if not os.path.exists(test_image_unzip_path):
        z = zipfile.ZipFile(test_image_zip_path, 'r')
        z.extractall(path=test_image_unzip_path)
        z.close()
        os.remove(test_image_zip_path)
    print("Unzip test imageas file done!")

    train_csv = os.path.join(unzip_root, "train.csv")
    test_csv = os.path.join(unzip_root, "test.csv")
    transform_list(train_image_zip_path, train_csv, "train_list.txt")
    print("Train_list: {:} done!".format("train_list.txt"))
    transform_list(test_image_zip_path, test_csv, "test_list.txt")
    print("Test_list: {:} done!".format("test_list.txt"))



if __name__ == '__main__':
    main()