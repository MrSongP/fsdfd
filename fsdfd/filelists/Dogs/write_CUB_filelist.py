import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random


data_path = '/root/Stanford Dogs/Images'
savedir = './'
dataset_list = ['base','val','novel']


folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
folder_list.sort()
label_dict = dict(zip(folder_list,range(0,len(folder_list))))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    folder_path = folder_path.replace('\\', '/')
    classfile_list_all.append( [ join(folder_path, cf).replace('\\', '/') for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
    random.shuffle(classfile_list_all[i])




for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if 'base' in dataset and len(classfile_list) >= 42:
            if (i%2 == 0):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'val' in dataset and len(classfile_list) >= 42:
            if (i%4 == 1):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
        if 'novel' in dataset and len(classfile_list) >= 42:
            if (i%4 == 3):
                file_list = file_list + classfile_list
                label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folder_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item  for item in file_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item  for item in label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell()-1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" %dataset)

