import json
import traceback
from ..base import Base
from tornado.web import HTTPError
from tornado.concurrent import run_on_executor
from io import BytesIO
from PIL import Image
import numpy as np
import cv2 as cv
from ...lab.detection_bright_spots import detection_bright_spots
class BreastCalcify(Base):
    cfg = {'debug': False,
           'threshold': 48,
           'blur_ksize': 3,
           'dilate_ksize': 3,
           'object_area': (11, 300, 0.50)}

    @run_on_executor(executor='thread_pool')
    def worker(self):
        try:
            tmp = self.request.files.get('images')
            img = Image.open(BytesIO(tmp[0]['body']))
            imgformat = img.format
            if img.mode != 'RGB':
              img = img.convert('RGB')
            img = cv.cvtColor(np.asarray(img, dtype='uint8'), cv.COLOR_RGB2BGR)
            logs, bbox = detection_bright_spots(img, 170, self.cfg)
            if 'debug' in self.request.arguments:
                res = img.copy()
                cv.putText(res, 'cnt: {}'.format(len(logs)), (5, 5), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                for num, x, y, w, h in logs:
                    cv.rectangle(res, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv.putText(res, '#{:.2f},{},{}'.format(num/(h*w), w, h), (x, y-15), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                res = Image.fromarray(cv.cvtColor(res, cv.COLOR_BGR2RGB))
                tmpfile = BytesIO()
                res.save(tmpfile, format=imgformat)
                res = tmpfile.getvalue()
                log = 'detection/BreastCalcify?debug:{}'.format('msg')
            else:
                res = json.dumps({'state': 0, 'result': logs})
                log = 'detection/BreastCalcify?log:{}'.format('msg')
        except Exception:
            raise HTTPError(400, traceback.format_exc())
        return res, log