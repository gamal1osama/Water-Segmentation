import numpy as np
import tensorflow as tf

def normalize_image(image):
    norm_img = np.zeros_like(image, dtype=np.float32)
    for band_idx in range(image.shape[2]):
        band = image[:, :, band_idx]
        min_val, max_val = np.min(band), np.max(band)
        if max_val > min_val:
            norm_img[:, :, band_idx] = (band - min_val) / (max_val - min_val)
    return norm_img

def preprocess_image(image):
    # Select bands 1,2,3,4,5,6,11 → (0-based: 1,2,3,4,5,6,11)
    indices = [1, 2, 3, 4, 5, 6, 11]
    selected = image[:, :, indices]
    return selected

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    total = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2. * intersection + smooth) / (total + smooth)

def iou(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)
