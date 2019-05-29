import tensorflow as tf
from keras import backend as K
import numpy as np

def tf_log10(x):
    numerator = tf.log(x)
    denominator=tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * tf_log10((max_pixel **2) / (K.mean(K.square(y_pred-y_true))))

def PSNR_TEST(y_true, y_pred):
    max_pixel=255.0
    return 10.0 * tf_log10((max_pixel **2) / (K.mean(K.square(y_pred-y_true))))

def scale_psnr(gt, pred, lowres, scale):
    gt=np.resize(gt,(gt.shape[1], gt.shape[2]))
    pred=np.resize(pred,(pred.shape[1], pred.shape[2]))
    lowres=np.resize(lowres, (lowres.shape[1], lowres.shape[2]))

    gt=np.array(gt)
    gt=gt[scale:-scale, scale:-scale]
    pred=np.array(pred)
    pred=pred[scale:-scale, scale:-scale]

    lowres=np.array(lowres)
    lowres=lowres[scale:-scale, scale:-scale]

    return gt, pred, lowres