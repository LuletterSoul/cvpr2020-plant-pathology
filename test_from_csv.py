# @Author: yican, yelanlan
# @Date: 2020-07-07 14:48:03
# @Last Modified by:   yican
# @Last Modified time: 2020-07-07 14:48:03
# Standard libraries
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
from utils import init_hparams, init_logger, load_test_data, seed_reproducer, load_data
import pandas as pd


def save_csv_report(report, output_dir, class_names):
    repdf = pd.DataFrame(report).transpose()
    repdf.insert(loc=0, column='class', value=class_names + ["accuracy", "macro avg", "weighted avg"])
    repdf.to_csv(output_dir, index=False,float_format='%.4f')

def save_csv_confusion_matrix(confusion_matrix, output_dir, class_names):
    cm = pd.DataFrame(confusion_matrix, columns=class_names)
    cm.insert(loc=0, column=' ', value=class_names)
    cm.to_csv(output_dir, index=False)



if __name__ == "__main__":
    # Init Hyperparameters
    # test_result_files= ['submission_distill_five.csv', 'submission_distill_five_no_aug.csv','submission_distill_one.csv','submission_distill_one_no_aug.csv']
    test_dir = 'test_results'
    test_result_files= os.listdir(test_dir)
    hparams = init_hparams()

    # Make experiment reproducible
    seed_reproducer(hparams.seed)

    tiemstamp = time.strftime("%Y%m%d-%H%M", time.localtime()) 

    output_dir = f'outputs/{tiemstamp}'
    # init logger
    logger = init_logger("kun_out", log_dir=hparams.log_dir)

    os.makedirs(output_dir,exist_ok=True)

    class_names = ['OK', 'AirRoomShake', 'Dead', 'Empty', 'NoAirRoom', 'Split', 'Weak']
    bn_class_names = ['OK', 'NoOK']
    # Load data
    test_data, data = load_test_data(logger, hparams.data_folder)
    for pred_file in test_result_files:
        pred_data = pd.read_csv(os.path.join(test_dir, pred_file))
        filename = os.path.splitext(pred_file)[0]

        gt_labels = test_data.iloc[:, 1:].to_numpy()
        pred_labels = pred_data.iloc[:, 1:].to_numpy()

        gt_labels = np.argmax(gt_labels, axis=1)
        pred_labels = np.argmax(pred_labels, axis=1)

        bn_gt_labels = gt_labels.copy()
        bn_pred_labels = pred_labels.copy()

        bn_gt_labels[bn_gt_labels!=0] = 1
        bn_pred_labels[bn_pred_labels!=0] = 1


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
