## require
```
wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh
bash Anaconda3-5.1.0-Linux-x86_64.sh
echo $'export PATH="/root/anaconda3/bin:$PATH"' >> /etc/profile
source /etc/profile
conda info
conda update conda

conda install tornado
conda install scikit-image
conda install -c menpo opencv3

pip install --upgrade tensorflow
pip install pyyaml h5py
pip install keras
```

## Service

- host: 116.62.198.34
- port: ?

开机启动:
```
vim /etc/rc.local

##port ?
cd /root/hej/object-masking
/root/anaconda3/bin/python3 server.py ? &
```

手动启动:
```
ps aux | grep ?
kill ?
cd /root/hej/object-masking
/root/anaconda3/bin/python3 server.py ? &
```

请求服务(Python3):
```
url = ?

vals = {'valName1': .., 'valName2': ..}

files = [('images', ('fileName1.png', open('1.png', 'rb'), 'image/png')),
         ('images', ('fileNmae2.png', open('2.png', 'rb'), 'image/png'))]

import requests
response = requests.post(url, data=vals, files=files)
print(response.status_code)
print(response.url)
print(response.text)

from io import BytesIO
from PIL import Image
Image.open(BytesIO(response.content))
```

`response.text`或`response.content`解读,见与之相关联的任务.

## {host}:{port}/detection/BreastCalcify
钼靶影像检测,乳房钙化

- 平均响应时长:1s(期望)
- 天均请求次数:300(期望)

inputs:
```
images: such as `<input type="file" ..`, named as `images`
```

result:
```
{
    "state": 0,
    "result": json string, such as `[(score, x, y, w, h), (score, x, y, w, h)]`
}
```

>注:任何非零的`state`标志着`result`和预期不同,比如填充为错误信息.

test(python):
```
import requests

debug = False

url = 'http://116.62.198.34:9001/detection/BreastCalcify?{}'.format('debug' if debug else '')
#url = 'http://127.0.0.1:9001/detection/BreastCalcify?{}'.format('debug' if debug else '')
files = [('images', ('pic1', open('1.jpg', 'rb'), 'image/jpeg')),]

response = requests.post(url, files=files)
print(response.status_code)
print(response.headers['content-type'])
print(response.url)

from io import BytesIO
from PIL import Image
Image.open(BytesIO(response.content)) if debug else response.text
```
