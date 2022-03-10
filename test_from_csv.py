# @Author: yican, yelanlan
# @Date: 2020-07-07 14:48:03
# @Last Modified by:   yican
# @Last Modified time: 2020-07-07 14:48:03
# Standard libraries
from pydoc import classname
import shutil
from unicodedata import name
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np
import time
import os

# Third party libraries
import torch
from scipy.special import softmax
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

# User defined libraries
from dataset import OpticalCandlingDataset, generate_transforms, PlantDataset
from train import CoolSystem
from utils import init_hparams, init_logger, load_test_data, load_test_data_with_header, seed_reproducer, load_data
import pandas as pd


def save_csv_report(report, output_dir, class_names):
    repdf = pd.DataFrame(report).transpose()
    repdf.insert(loc=0, column='class', value=class_names + ["accuracy", "macro avg", "weighted avg"])
    repdf.to_csv(output_dir, index=False,float_format='%.4f')

def save_csv_confusion_matrix(confusion_matrix, output_dir, class_names):
    cm = pd.DataFrame(confusion_matrix, columns=class_names)
    cm.insert(loc=0, column=' ', value=class_names)
    cm.to_csv(output_dir, index=False)


def save_false_negative(data_folder, pred_data, gt_data, class_names, output_dir):
    """select the false negative samples based on predictions and ground truth

    Args:
        data_folder (_type_): _description_
        pred_data (_type_): _description_
        gt_data (_type_): _description_
        class_names (_type_): _description_
        output_dir (_type_): _description_
    """
    for class_name in class_names[1:]:
        selected_pred = pred_data[gt_data[class_name] == 1]         
        false_positive = selected_pred[selected_pred['pred_class'] == 'OK']
        file_names = false_positive['filename']
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        for filename in file_names:
            file_path = os.path.join(data_folder, filename)
            shutil.copy(file_path, class_output_dir)
            print(f'Copied file {file_path} to {class_output_dir}')
        print(f'{class_name}: {false_positive.shape[0]}')
thresh = [0.9, 0.8, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
if __name__ == "__main__":
    # Init Hyperparameters
    # test_result_files= ['submission_distill_five.csv', 'submission_distill_five_no_aug.csv','submission_distill_one.csv','submission_distill_one_no_aug.csv']
    test_dir = 'test_results'
    test_result_files= [filename for filename in os.listdir(test_dir) if filename.endswith('.csv')]
    hparams = init_hparams()

    # Make experiment reproducible
    seed_reproducer(hparams.seed)

    timestamp = time.strftime("%Y%m%d-%H%M", time.localtime()) 

    output_dir = f'outputs/{timestamp}'

    fn_output_dir = os.path.join(output_dir, 'fn')

    # init logger
    logger = init_logger("kun_out", log_dir=hparams.log_dir)

    os.makedirs(output_dir,exist_ok=True)

    class_names = ['OK', 'AirRoomShake', 'Dead', 'Empty', 'NoAirRoom', 'Split', 'Weak', 'Flower']

    bn_class_names = ['OK', 'NoOK']

    header_names = ['filename'] + class_names 
    gt_data, data = load_test_data_with_header(logger, hparams.data_folder, header_names)
    # print(test_data[:10])
    for pred_file in test_result_files:
        pred_data = pd.read_csv(os.path.join(test_dir, pred_file), names=header_names)
        # Exclude first column (filename), and calculate the the prediction results, which is appended to the last column.
        pred_data['pred_class'] = pred_data.iloc[:, 1:].idxmax(axis=1)
        # print(pred_data[:10])
        filename = os.path.splitext(pred_file)[0]

        save_false_negative(hparams.data_folder, pred_data, gt_data,  class_names, fn_output_dir)

        # Read ground truth labels from test data.
        # The label format is one-hot vector.
        gt_labels = gt_data.iloc[:, 1:].to_numpy()
        pred_labels = pred_data.iloc[:, 1:-1].to_numpy()


        thresh_pred_labels = np.concatenate([pred_labels[:, [0]],pred_labels[:, 1:].sum(axis=1, keepdims=True)], axis = 1)

        # Convert one-hot to class label.
        gt_labels = np.argmax(gt_labels, axis=1)
        pred_labels = np.argmax(pred_labels, axis=1)

        # Convert to binary classification
        bn_gt_labels = gt_labels.copy()
        bn_pred_labels = pred_labels.copy()


        # All the labels of bad eggs are set to 1
        # 0 represents good eggs
        # 1 represents bad eggs
        bn_gt_labels[bn_gt_labels!=0] = 1
        bn_pred_labels[bn_pred_labels!=0] = 1



        for th in thresh:
            th_pred_labels = np.argmax((thresh_pred_labels > th).astype(int), axis=1)
            # th_pred_labels[th_pred_labels!=0] = 1
            th_output_dir = os.path.join(output_dir, 'th')
            os.makedirs(th_output_dir, exist_ok=True)
            th_confusion_matrix =metrics.confusion_matrix(bn_gt_labels, th_pred_labels)
            th_report = metrics.classification_report(bn_gt_labels, th_pred_labels, target_names=bn_class_names, output_dict=True)
            save_csv_report(th_report, os.path.join(th_output_dir,f'TH_Report_{filename}_th_{th}.csv'), bn_class_names)
            save_csv_confusion_matrix(th_confusion_matrix, os.path.join(th_output_dir,f'TH_CM_{filename}_th_{th}.csv'), bn_class_names)


        confusion_matrix =metrics.confusion_matrix(gt_labels, pred_labels)
        report = metrics.classification_report(gt_labels, pred_labels, target_names=class_names, output_dict=True)

        bn_confusion_matrix =metrics.confusion_matrix(bn_gt_labels, bn_pred_labels)
        bn_report = metrics.classification_report(bn_gt_labels, bn_pred_labels, target_names=bn_class_names, output_dict=True)

        save_csv_report(report, os.path.join(output_dir,f'Report_{filename}.csv'), class_names)
        save_csv_confusion_matrix(confusion_matrix, os.path.join(output_dir,f'CM_{filename}.csv'), class_names)

        save_csv_report(bn_report, os.path.join(output_dir,f'BN_Report_{filename}.csv'), bn_class_names)
        save_csv_confusion_matrix(bn_confusion_matrix, os.path.join(output_dir,f'BN_CM_{filename}.csv'), bn_class_names)

        # df = pd.DataFrame.from_dict(report)
        # df.to_csv(f'Report_{filename}.csv')
        # classification_report_csv(report, os.path.join('outputs',f'Report_{filename}.csv'))
