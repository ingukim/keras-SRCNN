import h5py
import numpy as np
import glob, os, re, scipy.io

patch_size=32

def transpose_data(data,label):
    data=np.transpose(data, (2,0,1))
    label=np.transpose(label, (2,0,1))
    return data, label

def reshape_data(data, label, patch_size):
    data=np.reshape(data,(data.shape[0], patch_size, patch_size,1))
    label=np.reshape(label, (label.shape[0], patch_size, patch_size,1))
    return data, label

def prepare_data(path):
    with h5py.File(path, 'r') as hf:
        data=np.array(hf.get('patch_all'))
        label=np.array(hf.get('label_all'))

    data, label = transpose_data(data, label)
    data, label = reshape_data(data, label, patch_size)
    return data, label


def get_img_list(test_path):
    I=glob.glob(os.path.join(test_path, '*'))
    I=[f for f in I if re.search("^\d+.mat$", os.path.basename(f))]
    train_list=[]
    for f in I:
        if os.path.exists(f):
            if os.path.exists(f[:-4] + "_2.mat"): train_list.append([f, f[:-4] + "_2.mat", 2])
            if os.path.exists(f[:-4] + "_3.mat"): train_list.append([f, f[:-4] + "_3.mat", 3])
            if os.path.exists(f[:-4] + "_4.mat"): train_list.append([f, f[:-4] + "_4.mat", 4])
    return train_list

def get_test_image(test_list, offset, batch_size):
    target_list=test_list[offset:offset+batch_size]
    input_list=[]
    gt_list=[]
    scale_list=[]
    for pair in target_list:
        mat_dict=scipy.io.loadmat(pair[1])
        input_img=None
        if "img_2" in mat_dict: input_img = mat_dict["img_2"]
        elif "img_3" in mat_dict: input_img = mat_dict["img_3"]
        elif "img_4" in mat_dict: input_img = mat_dict["img_4"]
        else: continue

        gt_img = scipy.io.loadmat(pair[0])['img_raw']

        input_list.append(input_img)
        gt_list.append(gt_img)
        scale_list.append(pair[2])
    return input_list, gt_list, scale_list

