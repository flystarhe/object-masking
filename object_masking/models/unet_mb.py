from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D
def get_unet(in_shape, k_size=3):
    '''
    Feature maps are concatenated along last axis (for tf backend)
    '''
    merge_axis = -1
    data = Input(shape=in_shape)
    conv1 = Convolution2D(filters=32, kernel_size=k_size, padding='same', activation='relu')(data)
    conv1 = Convolution2D(filters=32, kernel_size=k_size, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(pool1)
    conv2 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(pool2)
    conv3 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(pool3)
    conv4 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(pool4)

    up1 = UpSampling2D(size=(2, 2))(conv5)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(up1)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(conv6)
    merged1 = concatenate([conv4, conv6], axis=merge_axis)
    conv6 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(merged1)

    up2 = UpSampling2D(size=(2, 2))(conv6)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(up2)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(conv7)
    merged2 = concatenate([conv3, conv7], axis=merge_axis)
    conv7 = Convolution2D(filters=256, kernel_size=k_size, padding='same', activation='relu')(merged2)

    up3 = UpSampling2D(size=(2, 2))(conv7)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(up3)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(conv8)
    merged3 = concatenate([conv2, conv8], axis=merge_axis)
    conv8 = Convolution2D(filters=128, kernel_size=k_size, padding='same', activation='relu')(merged3)

    up4 = UpSampling2D(size=(2, 2))(conv8)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(up4)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(conv9)
    merged4 = concatenate([conv1, conv9], axis=merge_axis)
    conv9 = Convolution2D(filters=64, kernel_size=k_size, padding='same', activation='relu')(merged4)

    conv10 = Convolution2D(filters=1, kernel_size=k_size, padding='same', activation='sigmoid')(conv9)

    output = conv10
    model = Model(data, output)
    return model


import os
import shutil
import collections
import cv2
import numpy as np
def preprocess_mask(data_dir, mask_dir, im_shape=None):
    '''
    data_dir='/data2/datasets/slyx/tmps/mb1_base_may_80'
    mask_dir='/data2/datasets/slyx/tmps/mb1_base_may_80_mask'
    im_shape=None  # (width, height)
    preprocess_mask(data_dir, mask_dir, im_shape)
    '''
    save_dir = os.path.join(mask_dir, 'data')
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.system('mkdir -m 777 -p {}'.format(save_dir))

    data = collections.defaultdict(list)
    for filename in os.listdir(mask_dir):
        filepath = os.path.join(mask_dir, filename)
        if filepath.endswith(('.jpg','.png')):
            data[filename.split('_')[0]].append(filepath)

    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if filepath.endswith(('.jpg','.png')):
            img = np.zeros(cv2.imread(filepath, 0).shape)
            for tmp in data[filename.split('.')[0]]:
                img = img + cv2.imread(tmp, 0)
            img = np.clip(img, 0, 1) * 255
            if im_shape is not None:
                img = cv2.resize(img, im_shape, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(save_dir, filename), img)

    return save_dir, im_shape


import os
import cv2
import numpy as np
def load_data(data_dir, mask_dir, im_shape=None):
    '''
    im_shape: (width, height)
    '''
    X, Y = [], []
    for filename in os.listdir(data_dir):
        if not filename.endswith(('.jpg','.png')):
            continue

        img = cv2.imread(os.path.join(data_dir, filename), 0)
        mask = cv2.imread(os.path.join(mask_dir, filename), 0)

        if im_shape is not None:
            img = cv2.resize(img, im_shape, interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, im_shape, interpolation=cv2.INTER_NEAREST)

        img = np.expand_dims(img, -1)
        mask = np.expand_dims(mask, -1)

        X.append(img)
        Y.append(mask)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    X_mean, X_std = X.mean(), X.std()

    print('Tips: X.mean = {}, X.std = {}'.format(X_mean, X_std))

    return (X-X_mean)/X_std, Y.clip(0, 1), (X_mean, X_std)


import os
import shutil
import cv2
import numpy as np
def load_data_plus(data_dir, mask_dir, save_dir=None, ksize=None, stride=None, threshold=36):
    '''
    data_dir = '/data2/object-masking/datasets/mb1_base_may_80'
    mask_dir = '/data2/object-masking/datasets/mb1_base_may_80_mask/data'
    save_dir = '/data2/object-masking/datasets/mb1_base_may_80_dataset'
    load_data_plus(data_dir, mask_dir, save_dir)
    '''
    if save_dir:
        if os.path.isdir(save_dir):
            shutil.rmtree(save_dir)
        os.system('mkdir -m 777 -p {}'.format(save_dir))
        os.system('mkdir -m 777 -p {}'.format(os.path.join(save_dir, 'data')))
        os.system('mkdir -m 777 -p {}'.format(os.path.join(save_dir, 'mask')))

    if ksize is None:
        ksize = 128

    if stride is None:
        stride = ksize // 2

    if threshold > 1:
        threshold /= ksize ** 2

    X, Y, idx = [], [], 0
    for filename in os.listdir(data_dir):
        if not filename.endswith(('.jpg', '.png')):
            continue

        img = cv2.imread(os.path.join(data_dir, filename), 0)
        mask = cv2.imread(os.path.join(mask_dir, filename), 0)

        hs = set(tuple(range(0, img.shape[0]-ksize, stride)) + (img.shape[0]-ksize,))
        ws = set(tuple(range(0, img.shape[1]-ksize, stride)) + (img.shape[1]-ksize,))

        for h in hs:
            for w in ws:
                sub_img = img[h:h+ksize, w:w+ksize]
                sub_mask = mask[h:h+ksize, w:w+ksize]

                if (sub_mask>0).sum()/sub_mask.size < threshold:
                    continue

                idx += 1
                if save_dir:
                    cv2.imwrite(os.path.join(save_dir, 'data/img{}.jpg'.format(idx)), sub_img)
                    cv2.imwrite(os.path.join(save_dir, 'mask/img{}.jpg'.format(idx)), sub_mask)
                else:
                    sub_img = np.expand_dims(sub_img, -1)
                    sub_mask = np.expand_dims(sub_mask, -1)
                    X.append(sub_img)
                    Y.append(sub_mask)

    if save_dir:
        return save_dir, idx
    else:
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        X_mean, X_std = X.mean(), X.std()
        return (X-X_mean)/X_std, Y.clip(0, 1), (X_mean, X_std)