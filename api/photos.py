#!/usr/bin/env python
# encoding: utf-8
"""
@author: Shanda Lau 刘祥德
@license: (C) Copyright 2019-now, Node Supply Chain Manager Corporation Limited.
@contact: shandalaulv@gmail.com
@software: 
@file: categories.py
@time: 2022/2/14 15:28
@version 1.0
@desc:
"""
from flask_restplus import Namespace, Resource, reqparse
from werkzeug.datastructures import FileStorage
from service import *
from PIL import Image
import io
import cv2

api = Namespace('photos', description='Photo Transmition')
# create data storage directory
# os.makedirs(Config.CONTENT_DIRECTORY, exist_ok=True)
# os.makedirs(Config.CAST_DATA_DIR, exist_ok=True)

image_all = reqparse.RequestParser()
image_all.add_argument('page', default=1, type=int)
image_all.add_argument('size', default=50, type=int, required=False)
image_all.add_argument('category', default='', type=str, required=False)

image_upload = reqparse.RequestParser()
image_upload.add_argument('file', location='files',
                          type=FileStorage, required=True,
                          help='PNG or JPG photos')

@api.route('/')
class Photos(Resource):

    @api.expect(image_upload)
    def post(self):
        """ Creates an image """
        args = image_upload.parse_args()
        image = args['file']

        image_path = '/data/lxd/datasets/2021-12-12-Eggs/Weak/174409520_Egg3_(ruopei--ok)_L_0_cam3.bmp'
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        image = image.transpose(1, 0, 2)

        # pil_image = Image.open(io.BytesIO(image.read()))
        # image = pil_image.convert('RGB')
        # pil_image.save(path)

        submission = []
        for idx in range(len(input_entry)):
            input_entry[idx].put(image)
        
        for idx in range(len(output_entry)):
            res = output_entry[idx].get()
            submission.append(res)

        return assemble_result(submission)

