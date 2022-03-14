import pandas as pd
import os
import numpy as np
import csv
import random
import cv2
import pandas as pd

from utils.logs import load_test_data, load_training_data
input_dir = '/data/lxd/datasets/2022-03-02-Eggs'
output_dir = '/data/lxd/datasets/2022-03-02-Eggs'
os.makedirs(output_dir, exist_ok=True)
label_map = {
    'OK': 0,
    'AirRoomShake': 1,
    'Dead': 2,
    'Empty': 3,
    'NoAirRoom': 4,
    'Split': 5,
    'Weak': 6,
    'Flower': 7
}

class_dirs = os.listdir(input_dir)
one_hot_vector = [0 for _ in range(len(label_map.keys()))]


def write_csv(csv_instance, class_dir, image_ids):
    for image_id in image_ids:
        one_hot = one_hot_vector.copy()
        one_hot[label_map[class_dir]] = 1
        row_item = [image_id] + one_hot
        csv_instance.writerow(row_item)
        # print(f'Processing {image_id}')


def split():
    with open(os.path.join(output_dir, 'train_4_3.csv'), 'w',
            newline='') as train_file:
        with open(os.path.join(output_dir, 'test_4_1.csv'), 'w',
                newline='') as test_file:
            train_writer = csv.writer(train_file)
            test_writer = csv.writer(test_file)
            for class_dir in class_dirs:
                class_path = os.path.join(input_dir, class_dir)
                if os.path.isdir(class_path):
                    image_ids = [
                        os.path.join(class_dir, image_id)
                        for image_id in os.listdir(class_path) if image_id.endswith('.bmp')
                    ]
                    random.shuffle(image_ids)
                    train_image_ids = image_ids[:len(image_ids) // 4 * 3]
                    test_image_ids = image_ids[len(image_ids) // 4 * 3:]
                    write_csv(train_writer, class_dir, train_image_ids)
                    write_csv(test_writer, class_dir, test_image_ids)
                    print(f'{class_dir} - {len(test_image_ids)}')

def validate():
    train_data, _ = load_training_data(None, output_dir)
    test_data, _ = load_test_data(None, output_dir)

    train_filenames = train_data.iloc[:,0]
    test_filenames = test_data.iloc[:,0]

    error_train_filenames = []
    error_test_filenames = []
    # image = cv2.imread(os.path.join(input_dir, 'OK/000513851_Egg1_(ok)_R_0_cam2 - 副本.bmp'))
    # if image is None:
        # raise Exception()
    for filename in train_filenames:
        try:
            image_path = os.path.join(input_dir, filename)
            print(image_path)
            print(f'Processing {image_path}')
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f'Error {image_path}')
        except: 
            error_train_filenames.append(image_path)


    for filename in test_filenames:
        try:
            image_path = os.path.join(input_dir, filename)
            print(image_path)
            print(f'Processing {image_path}')
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f'Error {image_path}')
        except:
            error_test_filenames.append(image_path)
    print(error_train_filenames)
    print(error_test_filenames)
            
    
if __name__ == '__main__':
    # split()
    validate()