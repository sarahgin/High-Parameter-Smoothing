import numpy as np
import cv2
import glob
import tensorflow as tf
import re
import utils
from sklearn.feature_extraction import image as skimage


def get_sobel(im):
    sobelx = cv2.Sobel(im, cv2.CV_32F, 1, 0)
    sobely = cv2.Sobel(im, cv2.CV_32F, 0, 1)
    magnitude = np.sqrt(sobelx ** 2.0 + sobely ** 2.0)
    return magnitude


def get_canny(im):
    canny = cv2.Canny(im, 100, 200)
    return canny


def normImage(im):
    im_normalized = (im - np.amin(im)) / (np.amax(im) - np.amin(im))
    return im_normalized


def toGray(im):
    im_gray = np.mean(im, 2)
    return im_gray


def next_epoch(batch_size, data):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    num_of_batches = len(data)/batch_size
    indices_list = []
    start_idx = 0
    for i in range(int(num_of_batches)):
        indices = np.arange(start_idx, start_idx + batch_size)
        indices_to_append = [idx[j] for j in indices]
        indices_list.append(indices_to_append)
        start_idx = start_idx + batch_size
    return indices_list

def next_batch(indices, data, labels, sobels):
    data_shuffle = [data[i] for i in indices]
    sobel_shuffle = [sobels[k] for k in indices]
    labels_shuffle = [labels[j] for j in indices]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle), np.asarray(sobel_shuffle)


def loadSobels(input_path, full_image_rows, full_image_cols, patch_size, patch_stride, is_patches):
    image_list = []
    count = 1
    for filename in glob.glob(input_path + '*.jpg'):
        image = cv2.imread(filename)
        image = np.mean(image, axis=2)

        imaget = np.ndarray(shape=[full_image_rows,full_image_cols])
        if image.shape[0] < image.shape[1]:
            imaget = np.transpose(image)
            image = imaget

        image = normImage(image)
        image = np.float32(image)

        if is_patches:
            image_patches = skimage.extract_patches(image, (patch_size, patch_size), extraction_step=(patch_stride,patch_stride)).squeeze()
            dim1 = image_patches.shape[0]
            dim2 = image_patches.shape[1]
            for i in range(dim1):
                for j in range(dim2):
                    image_list.append(image_patches[i, j, :, :])
        else:
            image_list.append(image)

        count = count + 1

    return np.array(image_list)

def loadImages(input_path, full_image_rows, full_image_cols, patch_size, patch_stride, is_patches):
    image_list = []
    count = 1
    for filename in glob.glob(input_path + '*.jpg'):
        image = cv2.imread(filename)
        imaget = np.ndarray(shape=[full_image_rows,full_image_cols,3])
        if image.shape[0] < image.shape[1]:
            imaget[:, :, 0] = np.transpose(image[:, :, 0])
            imaget[:, :, 1] = np.transpose(image[:, :, 1])
            imaget[:, :, 2] = np.transpose(image[:, :, 2])
            image = imaget

        #filenameOnly = re.search("\\d*\\.jpg$", filename).group()
        #filenameOnly = filenameOnly.replace('.jpg', '')
        #image_for_canny = np.uint8(image)
        #canny = cv2.Canny(image_for_canny, 200, 300)
        #utils.save_image(canny, './BSD300/test-canny/' + filenameOnly + '.jpg')

        image = normImage(image)
        image = np.float32(image)
        if is_patches:
            image_patches = skimage.extract_patches(image, (patch_size, patch_size,3), extraction_step=(patch_stride,patch_stride,1)).squeeze()
            dim1 = image_patches.shape[0]
            dim2 = image_patches.shape[1]
            for i in range(dim1):
                for j in range(dim2):
                    image_list.append(image_patches[i, j, :, :, :])
        else:
            image_list.append(image)

        count = count + 1

    return np.array(image_list)

def loadGroundTruth(ground_truth_path, full_image_rows, full_image_cols, patch_size, patch_stride, is_patches):
    image_list = []
    count = 1
    for filename in glob.glob(ground_truth_path + '*.jpg'):
        image = cv2.imread(filename)
        image = np.mean(image, axis=2)

        imaget = np.ndarray(shape=[full_image_rows, full_image_cols])
        if image.shape[0] < image.shape[1]:
            imaget = np.transpose(image)
            image = imaget

        image = normImage(image)
        image = np.float32(image)
        if is_patches:
            image_patches = skimage.extract_patches(image, (patch_size, patch_size), extraction_step=(patch_stride, patch_stride))
            dim1 = image_patches.shape[0]
            dim2 = image_patches.shape[1]
            for i in range(dim1):
                for j in range(dim2):
                    image_list.append(image_patches[i, j, :, :])
        else:
            image_list.append(image)

        count = count + 1
    return np.array(image_list)

"""add one dim for batch"""
# VGG19 requires input dimension to be (batch, height, width, channel)
def add_one_dim(image):
    shape = (1,) + image.shape
    return np.reshape(image, shape)