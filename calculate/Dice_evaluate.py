import glob
import os
import SimpleITK as sitk
import numpy as np
import argparse

import pandas as pd
from medpy import metric
import re

def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())


def hd(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = metric.binary.hd95(pred, gt)
        return hd95
    else:
        return 0

def assert_shape(test, reference):

    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)


class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = int(((self.test != 0) * (self.reference != 0)).sum())
        self.fp = int(((self.test != 0) * (self.reference == 0)).sum())
        self.tn = int(((self.test == 0) * (self.reference == 0)).sum())
        self.fn = int(((self.test == 0) * (self.reference != 0)).sum())
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full


def dice_confusion(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(2. * tp / (2 * tp + fp + fn))

def jaccard(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp + fn))

def precision(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fp))


def sensitivity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TP / (TP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tp / (tp + fn))


def specificity(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """TN / (TN + FP)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if reference_full:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return float(tn / (tn + fp))

def process_label(label):

    pancreas = label == 1
    return pancreas

def test():
    label_path = r''
    label_list = sorted(glob.glob(os.path.join(label_path, '*nii.gz')))
    infer_list = sorted(glob.glob(os.path.join(r'', '*nii.gz')))
    print('loading success ----')
    print(f'number of labeled cases:{len(label_list)},'
          f'number of infered cases:{len(infer_list)}')

    Dice_pancreas = []
    HD_pancreas = []
    jaccard_pancreas = []
    precision_pancreas = []
    sensitivity_pancreas = []
    vol_file = os.path.join(label_path,'complete_val')

    if not os.path.exists(vol_file):
        os.makedirs(vol_file)


    for i, label_path in enumerate(label_list):
        #------------------------------------------------------------------------------find the corresponding file
        # name = label_path.split('/')[-1]
        # print(f'calculating {name}')
        # regex = re.compile(r'\d\d\d+')
        # num_label = max(regex.findall(label_path.split('\\')[-1]))
        # infer_name = num_label + '.nii.gz'
        # infer_path = [i for i in infer_list if infer_name in i  ]

        #------------------------------------------------------------------------------read nii img
        # -----------------------------------------------------------------------------------------
        label_path = label_list[i]
        infer_path = infer_list[i]
        print(f'is doing {i}')
        # ------------------------------------------------------------------------------------------
        label = read_nii(label_path)
        infer = read_nii(infer_path).squeeze()

        label_pancreas = process_label(label)
        infer_pancreas = process_label(infer)


        confusion_matrix = ConfusionMatrix(label_pancreas, infer_pancreas)
        jaccard_pancreas.append(jaccard(confusion_matrix=confusion_matrix))
        precision_pancreas.append(precision(confusion_matrix=confusion_matrix))
        sensitivity_pancreas.append(sensitivity(confusion_matrix=confusion_matrix))

        Dice_pancreas.append(dice(infer_pancreas, label_pancreas))
        HD_pancreas.append(hd(infer_pancreas, label_pancreas))

    data = {
        'Dice' : pd.Series(Dice_pancreas),
        'HD95' : pd.Series(HD_pancreas),
        'Jaccard' : pd.Series(jaccard_pancreas),
        'Precision' : pd.Series(precision_pancreas),
        'Sensitivity' : pd.Series(sensitivity_pancreas),

    }
    df = pd.DataFrame(data)
    df.to_excel(os.path.join(vol_file, 'evaluate.xlsx'))

if __name__ == '__main__':
    test()
