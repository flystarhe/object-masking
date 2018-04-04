import sys
sys.path.insert(0, '/data2/gits')


from pathlib import Path
from pyhej.image import dicom, arr2img, img2arr
from pyhej.utils import get_fname get_path_relative
from pyhej.utils import set_dir, set_parent, folder_split


data_dir = Path('/data2/datasets/slyx/breast_lump')
save_dir = Path('/data2/outputs/hej_breast_lump')

set_dir(save_dir, rm=True)


cnt, logs = 0, []
for item in data_dir.glob('**/*'):
    plan = dicom.dcmread(str(item))
    if plan:
        cnt += 1
        data = dicom.toimage(plan, RescaleType='MUBA')
        name = get_fname(get_path_relative(item, data_dir)) + '.jpg'
        arr2img(data).save(str(save_dir/name))
        logs.append((cnt, name))
print('logs:\n' + '\n  '.join(str(i) for i in logs[:3]+logs[-3:]))
logs = folder_split(save_dir, size=100, pattern='*.jpg')
print('logs:\n' + '\n  '.join(str(i) for i in logs[:3]+logs[-3:]))