def be_merge(box1, box2, maxGap=1):
    num1, x1, y1, w1, h1 = box1
    num2, x2, y2, w2, h2 = box2

    x, y = min(x1, x2), min(y1, y2)
    w, h = max(x1+w1, x2+w2) - x, max(y1+h1, y2+h2) - y

    if w - w1 - w2 > maxGap:
        return False

    if h - h1 - h2 > maxGap:
        return False

    return True


def to_merge(box1, box2):
    num1, x1, y1, w1, h1 = box1
    num2, x2, y2, w2, h2 = box2

    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1+w1, x2+w2) - x
    h = max(y1+h1, y2+h2) - y

    return num1+num2, x, y, w, h


def todo_logs(logs, resx=[]):
    items = logs.copy()
    times = range(2*len(logs))
    for i in times:
        box1 = items.pop()
        for box2 in items:
            if be_merge(box1, box2, 3):
                break
            else:
                box2 = None
        if box2 is None:
            items.append(box1)
        else:
            items.remove(box2)
            items.append(to_merge(box1, box2))
    return items


import cv2
def subs_label(bins, ksize=3, threshold=None):
    if threshold is None:
        threshold = (ksize**2)//2
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
def detection_bright_spots(image, bright=200, cfg={}):
    debug = cfg.get('debug', False)
    threshold = cfg.get('threshold', 48)
    blur_ksize = cfg.get('blur_ksize', 3)
    dilate_ksize = cfg.get('dilate_ksize', 3)
    kmin, kmax, rate = cfg.get('object_area', (9, 100, 0.30))

    if isinstance(image, str):
        image = cv2.imread(image)

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

    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    grad = (grad_x+grad_y>threshold).astype(np.uint8)
    kernel = np.ones((dilate_ksize, dilate_ksize), np.uint8)
    grad = cv2.dilate(grad, kernel, iterations=1)

    grad = np.logical_and(subs_label(grad, 3), cv2.medianBlur(gray, blur_ksize)>bright).astype('uint8')

    logs = area_label(grad, kmin, kmax, rate, bbox)

    if debug:
        return logs, bbox, grad
    else:
        return logs, bbox


import os
import cv2
import numpy as np
def save_image(image, logs, bbox=None, grad=None, save_dir=None):
    if isinstance(image, str):
        image = cv2.imread(image)
    source = image
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
        save_path = os.path.join(save_dir, '{}.jpg'.format(len(os.listdir(save_dir))))
        cv2.imwrite(save_path, target)
        return save_path