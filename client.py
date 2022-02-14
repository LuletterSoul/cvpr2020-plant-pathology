#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software:
@file: __init__.py.py
@time: 2022/2/13 19:39
@version 1.0
@descwerkzeug:
"""


from tkinter.tix import Tree
import requests
import os
import time
import cv2


test_file_path= '/Users/shandalau/Documents/2022-02-13-Eggs-Test'
ourput_dir = '/Users/shandalau/Documents/2022-02-13-Eggs-Test-Result'
os.makedirs(ourput_dir, exist_ok=True)
filenames = os.listdir(test_file_path)

num_samples = len(filenames)

start = time.time()
for filename in filenames:
    s = time.time()
    # image_path = '/Users/shandalau/Downloads/180052068_Egg4_(sipei)_L_0_cam3.bmp'
    image_path = os.path.join(test_file_path, filename)
    image = cv2.imread(image_path)
    api = 'http://10.8.0.94:5000/api/photos/'
    # api = 'http://221.226.81.54:1211/api/photos/'
    response = requests.post(api, files = {"file": open(image_path, 'rb')})
    pred = response.json()
    label = filename[:-4].split('_')[-4][1:-1]
    image = cv2.putText(image, f'Label: {label}', (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    image = cv2.putText(image, f'Pred: {pred}', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('Classification Result',image)
    cv2.waitKey(1)
    cv2.imwrite(os.path.join(ourput_dir, filename), image)
    print(response.json())
    e = time.time()
    print(f'Processing time {round(e - s, 2)} (s)')

end = time.time()
total = end - start
print(f'{round(total / num_samples, 2)} (s), {round(num_samples / total, 2)} (fps)')


