'''github
https://github.com/imlab-uiip/lung-segmentation-2d
'''


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
import cv2
import numpy as np
def preprocess(data_dir, save_dir, im_shape=None):
    '''
    data_dir='/data2/object-masking/datasets/tmps/MontgomerySet'
    save_dir='/data2/object-masking/datasets/tmps/mask'
    im_shape=(512, 512)  # (width, height)
    preprocess(data_dir, save_dir, im_shape)
    '''
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)

    os.system('mkdir -m 777 -p {}'.format(save_dir))
    os.system('mkdir -m 777 -p {}'.format(os.path.join(save_dir, 'imgs')))
    os.system('mkdir -m 777 -p {}'.format(os.path.join(save_dir, 'mask')))

    dir_img = os.path.join(data_dir, 'CXR_png')
    dir_left = os.path.join(data_dir, 'ManualMask', 'leftMask')
    dir_right = os.path.join(data_dir, 'ManualMask', 'rightMask')

    for i, filename in enumerate(os.listdir(dir_img), 1):
        path_img = os.path.join(dir_img, filename)
        path_left = os.path.join(dir_left, filename)
        path_right = os.path.join(dir_right, filename)

        if os.path.isfile(path_left) and os.path.isfile(path_right):
            img = cv2.imread(path_img, 0)
            left = cv2.imread(path_left, 0)
            right = cv2.imread(path_right, 0)

            if im_shape is not None:
                img = cv2.resize(img, im_shape, interpolation=cv2.INTER_NEAREST)
                left = cv2.resize(left, im_shape, interpolation=cv2.INTER_NEAREST)
                right = cv2.resize(right, im_shape, interpolation=cv2.INTER_NEAREST)

            cv2.imwrite(os.path.join(save_dir, 'imgs/{:04d}.png'.format(i)), img)
            cv2.imwrite(os.path.join(save_dir, 'mask/{:04d}.png'.format(i)), np.clip(left + right, 0, 255))


import os
import cv2
import numpy as np
def load_data(data, im_shape=None):
    '''
    data: str or list [(raw, mask), ..]
    im_shape: (width, height)
    '''
    if isinstance(data, str):
        path, data = data, []
        for i in os.listdir(os.path.join(path, 'imgs')):
            if i.endswith(('png', 'jpg', 'jpeg')):
                data.append(tuple(os.path.join(path, j, i) for j in ('imgs', 'mask')))

    X, y = [], []
    for img, mask in data:
        img = cv2.imread(img, 0)
        mask = cv2.imread(mask, 0)
        if im_shape is not None:
            img = cv2.resize(img, im_shape, interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, im_shape, interpolation=cv2.INTER_NEAREST)
        img = np.expand_dims(img, -1)
        mask = np.expand_dims(mask, -1)

        X.append(img)
        y.append(mask)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    X_mean, X_std = X.mean(), X.std()

    print('Tips: X.mean = {}, X.std = {}'.format(X_mean, X_std))

    return (X-X_mean)/X_std, y.clip(0, 1)


import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
def training(data=None, im_shape=(256, 256), epochs=3):
    if data is None:
        data = '/data2/object-masking/datasets/tmps/mask'

    X, y = load_data(data, im_shape)
    X_train, y_train = X[10:], y[10:]
    X_val, y_val = X[:10], y[:10]

    in_shape = X[0].shape
    model = get_unet(in_shape)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(model, 'train_unet.png', show_shapes=True)

    ################################################################
    if not os.path.isdir('tmps'):
        os.system('mkdir -m 777 -p {}'.format('tmps'))

    checkpoint = ModelCheckpoint(filepath='tmps/unet.checkpoint', verbose=0)
    best_model = ModelCheckpoint(filepath='tmps/unet.best.model', verbose=0, save_best_only=True)

    model.fit(X_train, y_train, batch_size=8, epochs=epochs, callbacks=[checkpoint, best_model], validation_split=0.1)

    return model.evaluate(X_val, y_val, batch_size=8, verbose=0)


import cv2
import numpy as np
from keras.models import load_model
def testing(model, data, im_shape=None, mean_and_std=None):
    '''
    im_shape: (width, height)
    mean_and_std: (X_mean, X_std)
    '''
    if isinstance(model, str):
        model = load_model(filepath=model)

    res = []
    for img in data:
        if isinstance(img, str):
            img = cv2.imread(img, 0)

        if im_shape is not None:
            img = cv2.resize(img, im_shape, interpolation=cv2.INTER_NEAREST)

        x = img
        if mean_and_std is not None:
            x = x - mean_and_std[0]
            x = x / mean_and_std[1]

        if x.ndim == 2:
            x = np.expand_dims(x, -1)

        y = model.predict(np.asarray([x]))

        res.append((img, y[0]))

    return res


import numpy as np
from skimage import morphology
def remove_small_regions(x, im_shape, rate=0.02):
    '''Morphologically removes small (less than size) connected regions of 0s or 1s
    x: ndarray, int or bool type
    im_shape: (width, height)
    '''
    x = x.copy()
    size = int(rate * np.prod(im_shape))
    morphology.remove_small_objects(x, min_size=size, in_place=True)
    morphology.remove_small_holes(x, min_size=size, in_place=True)
    return x


import numpy as np
from skimage import morphology, color
def image_mask(img, pred, mask=None, alpha=1):
    '''
    img: ndarray, opencv imread grayscale
    pred: ndarray, int or bool type
    Returns image with:
        predicted lung field filled with blue
        GT lung field outlined with red
    '''
    img_color = np.dstack((img, img, img))
    out_color = np.zeros(img.shape + (3,))
    boundary = morphology.dilation(pred, morphology.disk(3)) - pred
    out_color[boundary == 1] = [1, 0, 0]

    img_hsv = color.rgb2hsv(img_color)
    out_hsv = color.rgb2hsv(out_color)

    img_hsv[..., 0] = out_hsv[..., 0]
    img_hsv[..., 1] = out_hsv[..., 1] * alpha

    return color.hsv2rgb(img_hsv)


def IoU(y_true, y_pred):
    '''
    Returns Intersection over Union score for ground truth and predicted masks
    '''
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1.) / (union + 1.)


def Dice(y_true, y_pred):
    '''
    Returns Dice Similarity Coefficient for ground truth and predicted masks
    '''
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)