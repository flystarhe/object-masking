import requests
from pathlib import Path
from io import BytesIO
from PIL import Image


debug = True


url = 'http://116.62.198.34:9001/detection/BreastCalcify?{}'.format('debug' if debug else '')


data_dir = Path('/data2/datasets/slyx/tmps/mb1_base_may_80')
save_dir = Path('/data2/outputs/tmp_20180314')


for item in data_dir.glob('*.jpg'):
    files = [('images', ('pic', open(item.as_posix(), 'rb'), 'image/jpeg')),]
    response = requests.post(url, files=files)
    if response.status_code == 200:
        filename = str(save_dir/(item.name[:-4]+'_.jpg'))
        Image.open(BytesIO(response.content)).save(filename)
    else:
        print(response.status_code, item.name)