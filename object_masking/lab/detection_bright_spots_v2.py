#import sys
#sys.path.insert(0, '/data2/gits')


import cv2
def subs_label(bins, ksize=7, threshold=None):
    if threshold is None:
        threshold = (ksize//2)**2
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.filter2D(bins, -1, kernel) > threshold


import cv2
import numpy as np
from skimage import measure
def area_label(gray, kmin=10, kmax=100, rate=0.4, bbox=(0, 0, 0, 0)):
    logs = []
    labels = measure.label(gray, neighbors=8, background=0)
    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(gray.shape, dtype='uint8')
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        x, y, w, h = cv2.boundingRect(labelMask)
        if 0.5 < h/w < 2.0 and kmin < (h+w)/2 < kmax and rate < numPixels/(h*w):
            logs.append((numPixels, x+bbox[0], y+bbox[1], w, h))
    return logs


import cv2
import numpy as np
import scipy.signal as signal
def detection_bright_spots(filename, bright=200, cfg={}):
    debug = cfg.get('debug', False)
    threshold = cfg.get('threshold', 13)
    blur_ksize = cfg.get('blur_ksize', 3)
    dilate_ksize = cfg.get('dilate_ksize', 3)
    kmin, kmax, rate = cfg.get('object_area', (7, 100, 0.40))

    image = cv2.imread(filename)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.boundingRect((gray > bright).astype('uint8'))
    x = max(0, x-30)
    y = max(0, y-30)
    w = min(gray.shape[1], w+30)
    h = min(gray.shape[0], h+30)
    bbox = (x, y, w, h)
    gray = gray[y:y+h, x:x+w]

    #gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    #gray = cv2.medianBlur(gray, blur_ksize)
    gray = gray.astype(np.uint8)

    grad = -cv2.Laplacian(gray, cv2.CV_64F)
    grad = np.logical_and(grad>threshold, gray>bright).astype(np.uint8)
    resx = grad.copy()

    grad = subs_label(grad, 7).astype('uint8')
    kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
    grad = cv2.dilate(grad, kernel, iterations=1)

    logs = area_label(grad, kmin, kmax, rate, bbox)

    if debug:
        return logs, bbox, grad
    else:
        return logs, bbox


import os
import cv2
import numpy as np
def save_image(filename, logs, bbox=None, grad=None, save_dir=None):
    source = cv2.imread(filename)
    masked = source.copy()
    for numPixels, x, y, w, h in logs:
        cv2.rectangle(masked, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(masked, '#{:.2f},{},{}'.format(numPixels/(h*w), w, h), (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    if bbox is not None:
        x, y, w, h = bbox
        source = source[y:y+h, x:x+w]
        masked = masked[y:y+h, x:x+w]
    if grad is None:
        target = np.hstack((source, masked))
    else:
        if np.max(grad) <= 1:
            grad *= 255
        temp = np.zeros(source.shape[:2])
        temp[:grad.shape[0], :grad.shape[1]] = grad
        grad = temp
        target = np.dstack((grad, grad, grad))
        target = np.hstack((source, masked, target))
    if save_dir is None:
        return target
    else:
        if not os.path.isdir(save_dir):
            os.system('mkdir -m 777 -p {}'.format(save_dir))
        save_path = os.path.join(save_dir, os.path.basename(filename))
        cv2.imwrite(save_path, target)
        return save_path


import os
import shutil
bright = 200
cfg = {'debug': True,
       'threshold': 13,
       'blur_ksize': 3,
       'dilate_ksize': 3,
       'object_area': (7, 100, 0.40)}
data_dir = '/data2/datasets/slyx/tmps/mb1_base_jun'
for i in range(10):
    cfg['threshold'] = 3 + i * 3
    save_dir = '/data2/tmps/mb1_stage3/v2_{}_3_3_7_100_040'.format(cfg['threshold'])
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    os.system('mkdir -m 777 -p {}'.format(save_dir))
    filepaths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
    for filepath in filepaths:
        logs, bbox, grad = detection_bright_spots(filepath, bright, cfg)
        save_image(filepath, logs, bbox, grad, save_dir=save_dir)