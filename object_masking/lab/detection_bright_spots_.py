#import sys
#sys.path.insert(0, '/data2/gits')
from detection_bright_spots import *


import os
import shutil
cfg = {'debug': True,
    'threshold': 48,
    'blur_ksize': 3,
    'dilate_ksize': 3,
    'object_area': (11, 300, 0.50)}
data_dir = '/data2/datasets/slyx/tmps/mb1_base_jun'
save_dir = '/data2/tmps/output_mb1_base_jun'
if os.path.isdir(save_dir):
    shutil.rmtree(save_dir)
os.system('mkdir -m 777 -p {}'.format(save_dir))
filepaths = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir)]
for filepath in filepaths:
    image = cv2.imread(filepath)
    logs, bbox, grad = detection_bright_spots(image.copy(), 170, cfg)
    save_image(image.copy(), todo_logs(logs.copy()), bbox, grad, save_dir=save_dir)