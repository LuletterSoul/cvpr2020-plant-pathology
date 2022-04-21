import shutil
import numpy as np
import os
from sklearn import metrics
import pandas as pd
from .constant import *


def save_csv_report(report, output_dir, class_names, pos_thresh=0.50):
    """
    class precision recall f1-score support
    OK      ...      ...      ...      ...
    NoOK    ...      ...      ...      ...
    accuracy
    macro
    weighted 

    Args:
        report (_type_): _description_
        output_dir (_type_): _description_
        class_names (_type_): _description_

    Returns:
        _type_: _description_
    """
    repdf = pd.DataFrame(report).transpose()
    repdf.insert(loc=0,
                 column='class',
                 value=class_names + ["accuracy", "macro avg", "weighted avg"])
    repdf['pos_thresh'] = pos_thresh
    repdf.to_csv(output_dir, index=False, float_format='%.4f')
    return repdf


def save_csv_confusion_matrix(confusion_matrix, output_dir, class_names):
    cm = pd.DataFrame(confusion_matrix, columns=class_names)
    cm.insert(loc=0, column=' ', value=class_names)
    cm.to_csv(output_dir, index=False)
    return cm


def save_false_negative(data_folder, pred_data, gt_data, class_names,
                        output_dir):
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


def generate_report(pred_data, gt_data, pred_filename, output_dir):
    fn_output_dir = os.path.join(output_dir, 'fn')
    os.makedirs(fn_output_dir, exist_ok=True)
    # Exclude first column (filename), and calculate the the prediction results, which is appended to the last column.
    pred_data = pred_data.copy()
    gt_data = gt_data.copy()
    pred_data['pred_class'] = pred_data.iloc[:, 1:].idxmax(axis=1)
    # print(pred_data[:10])
    # pred_filename = os.path.splitext(pred_file)[0]

    # save_false_negative(hparams.data_folder, pred_data, gt_data,  class_names, fn_output_dir)

    # Read ground truth labels from test data.
    # The label format is one-hot vector.
    gt_labels = gt_data.iloc[:, 1:].to_numpy()
    pred_labels = pred_data.iloc[:, 1:-1].to_numpy()

    # N * 2
    thresh_pred_labels = np.concatenate(
        [pred_labels[:, [0]], pred_labels[:, 1:].sum(axis=1, keepdims=True)],
        axis=1)

    # Convert one-hot to class label.
    gt_labels = np.argmax(gt_labels, axis=1)
    pred_labels = np.argmax(pred_labels, axis=1)

    # Convert to binary classification
    bn_gt_labels = gt_labels.copy()
    bn_pred_labels = pred_labels.copy()

    # All the labels of bad eggs are set to 1
    # 0 represents good eggs
    # 1 represents bad eggs
    bn_gt_labels[bn_gt_labels != 0] = 1
    bn_pred_labels[bn_pred_labels != 0] = 1

    bn_reports = []
    print(f'Processing binary classification report')
    for th in THRESH:
        print(f'Report by positive threshold {th}')
        # N
        # try:
        # th_pred_labels = np.argmax((thresh_pred_labels > th).astype(int), axis=1)
        th_pred_labels = (~(thresh_pred_labels[:, 0] > th)).astype(int)
        # th_pred_labels[th_pred_labels!=0] = 1
        th_output_dir = os.path.join(output_dir, 'th')
        os.makedirs(th_output_dir, exist_ok=True)
        th_confusion_matrix = metrics.confusion_matrix(bn_gt_labels,
                                                       th_pred_labels)
        # print(bn_gt_labels)
        # print(th_pred_labels)
        th_report = metrics.classification_report(bn_gt_labels,
                                                  th_pred_labels,
                                                  target_names=BN_CLASS_NAMES,
                                                  output_dict=True)
        # th_report['pos_threshold'] = th
        bn_report_path = os.path.join(
            th_output_dir, f'TH_Report_{pred_filename}_th_{th}.csv')
        bn_report = save_csv_report(th_report,
                                    bn_report_path,
                                    BN_CLASS_NAMES,
                                    pos_thresh=th)
        bn_reports.append(bn_report)
        print(f'Report by positive threhold {th} saved into {bn_report_path}')
        # bn_reports['pos_threshold'] = th
        save_csv_confusion_matrix(
            th_confusion_matrix,
            os.path.join(th_output_dir, f'TH_CM_{pred_filename}_th_{th}.csv'),
            BN_CLASS_NAMES)
        # except Exception as e:
        # print(f'Error while handling {th}')
        # traceback.print_exc()
    bn_reports: pd.DataFrame = pd.concat(bn_reports).sort_values(
        by=['class', 'pos_thresh'], ascending=False)
    bn_reports.to_csv(os.path.join(output_dir,
                                   f'TH_Report_summary_{pred_filename}.csv'),
                      index=False,
                      float_format='%.4f')

    confusion_matrix = metrics.confusion_matrix(gt_labels, pred_labels)
    report = metrics.classification_report(gt_labels,
                                           pred_labels,
                                           target_names=CLASS_NAMES,
                                           output_dict=True)

    bn_confusion_matrix = metrics.confusion_matrix(bn_gt_labels,
                                                   bn_pred_labels)
    bn_report = metrics.classification_report(bn_gt_labels,
                                              bn_pred_labels,
                                              target_names=BN_CLASS_NAMES,
                                              output_dict=True)

    save_csv_report(report,
                    os.path.join(output_dir, f'Report_{pred_filename}.csv'),
                    CLASS_NAMES)
    save_csv_confusion_matrix(
        confusion_matrix, os.path.join(output_dir, f'CM_{pred_filename}.csv'),
        CLASS_NAMES)

    save_csv_report(bn_report,
                    os.path.join(output_dir, f'BN_Report_{pred_filename}.csv'),
                    BN_CLASS_NAMES)
    save_csv_confusion_matrix(
        bn_confusion_matrix,
        os.path.join(output_dir, f'BN_CM_{pred_filename}.csv'), BN_CLASS_NAMES)

    # df = pd.DataFrame.from_dict(report)
    # df.to_csv(f'Report_{filename}.csv')
    # classification_report_csv(report, os.path.join('outputs',f'Report_{filename}.csv'))
