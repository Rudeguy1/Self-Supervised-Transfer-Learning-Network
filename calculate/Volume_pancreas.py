import glob
import re

import SimpleITK as sitk
import os
import pandas as pd
import csv
import numpy as np



def volume_static(image, label, num):
    #-------------------------------------------------------------evaluate pancreas volum
    stats = sitk.StatisticsImageFilter()
    label = sitk.ReadImage(label)

    stats.Execute(label)
    spacing = label.GetSpacing()
    voxel = np.prod(spacing) * 0.001
    label_array = sitk.GetArrayFromImage(label)
    pancrese_num = label_array.copy()
    pancrese_num[label_array == 1] = 1

    sum_num = np.sum(pancrese_num)
    vol_pancreses = voxel * np.sum(pancrese_num)

    #------------------------------------------------------------evaluate- 190 < CT < -20 pancrease
    img = sitk.ReadImage(image)
    spacing_img = img.GetSpacing()
    voxel_img = np.prod(spacing_img) * 0.001
    if spacing != spacing_img:
        exit(1)
    img_array = sitk.GetArrayFromImage(img)
    fusion_img = img_array * label_array
    pancreseCT_num = fusion_img.copy()
    pancreseCT_num[pancreseCT_num > -20] = 0

    pancreseCT_num[pancreseCT_num < -190] = 0

    pancreseCT_num[pancreseCT_num <= -20] = 1



    #----------------------------------------------------------generate fat nii.gz files
    # pancreseCT_num_change = pancreseCT_num
    # pancreseCT_num_change[pancreseCT_num_change ==1] = 2
    # pancreseCT_num_nii = sitk.GetImageFromArray(pancreseCT_num_change)
    # pancreseCT_num_nii.SetSpacing(label.GetSpacing())
    # pancreseCT_num_nii.SetDirection(label.GetDirection())
    # pancreseCT_num_nii.SetOrigin(label.GetOrigin())
    # out_path = r'E:\data\enddata\attention_Swin_output_foldall5_endFat'
    # sitk.WriteImage(pancreseCT_num_nii, os.path.join(out_path, num + '.nii.gz'))
    #-------------------------------------------------------------------------------------
    vol_pancreses_CT = voxel_img * np.sum(pancreseCT_num)
    return vol_pancreses, vol_pancreses_CT, sum_num



Plain_img_path = r''
label_path = r''

img_list = sorted(glob.glob(os.path.join(Plain_img_path, '*nii.gz')))
label_list = sorted(glob.glob(os.path.join(label_path, '*nii.gz')))

all_vol_pamcreses = []
all_vol_pancreses_CT = []
ration = []
plain_num_all = []
num = []


for plain_img in img_list:
    plain_img_name = plain_img.split('\\')[-1]
    regix = re.compile(r'\d+')
    plain_num = str(max(regix.findall(plain_img_name))).zfill(4)
    label = os.path.join(label_path, plain_num + '.nii.gz')
    print(f'plain_num is {plain_num},label is {label}')
    vol_pancreses, vol_pancreses_CT, sum_num = volume_static(plain_img, label, num=plain_num)
    all_vol_pamcreses.append(vol_pancreses)
    all_vol_pancreses_CT.append(vol_pancreses_CT)
    result = vol_pancreses_CT / vol_pancreses
    ration.append(result)
    plain_num_all.append(plain_num)
    num.append(sum_num)

data = {'num' : pd.Series(plain_num_all),
        'vol_pancreas' : pd.Series(all_vol_pamcreses),
        'vol_pancreasFat_CT': pd.Series(all_vol_pancreses_CT),
        'ration': pd.Series(ration),
        'sum': pd.Series(num)
}
df = pd.DataFrame(data)
df.to_csv(os.path.join(label_path, 'valume_fat.csv'))
