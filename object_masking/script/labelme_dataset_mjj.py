import sys
sys.path.insert(0, '/data2/gits')


from pathlib import Path
from pyhej.image import dicom, arr2img, img2arr
from pyhej.utils import set_dir, set_parent, folder_split


data_dir = Path('/data2/datasets/slyx/mjj_20180207/cd2/DICOM')
save_dir = Path('/data2/outputs/mjj_20180207_cd2')
set_dir(save_dir, rm=True)


cnt, logs = 0, []
for item in data_dir.glob('**/*'):
    plan = dicom.dcmread(str(item))
    if plan:
        cnt += 1
        data = dicom.toimage(plan, RescaleType=None)
        name = '{:06d}-'.format(cnt) + '-'.join(item.relative_to(data_dir).parts) + '.jpg'
        arr2img(data).save(str(save_dir/name))
        logs.append((cnt, name))
print('logs:\n' + '\n  '.join(str(i) for i in logs[:3]+logs[-3:]))
logs = folder_split(save_dir, size=100, pattern='*.jpg')
print('logs:\n' + '\n  '.join(str(i) for i in logs[:3]+logs[-3:]))