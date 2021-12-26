import pandas as pd
import os
import numpy as np
import csv
import random

input_dir = '/data/lxd/datasets/2021-12-12-Eggs'
output_dir = '/data/lxd/datasets/2021-12-12-Eggs'

label_map = {
    'OK': 0,
    'AirRoomShake': 1,
    'Dead': 2,
    'Empty': 3,
    'NoAirRoom': 4,
    'Split': 5,
    'Weak': 6
}

class_dirs = os.listdir(input_dir)
one_hot_vector = [0 for i in range(7)]


def write_csv(csv_instance, class_dir, image_ids):
    for image_id in image_ids:
        one_hot = one_hot_vector.copy()
        one_hot[label_map[class_dir]] = 1
        row_item = [image_id] + one_hot
        csv_instance.writerow(row_item)
        print(f'Processing {image_id}')


with open(os.path.join(output_dir, 'train_4_3.csv'), 'a',
          newline='') as train_file:
    with open(os.path.join(output_dir, 'test_4_1.csv'), 'a',
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