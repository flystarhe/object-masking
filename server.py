import sys
import tornado.ioloop
import tornado.web
from object_masking.web.worker import Guess
from object_masking.web.detection import BreastCalcify

router = [(r'/', Guess),
          (r'/detection/BreastCalcify', BreastCalcify)]

if __name__ == '__main__':
    app = tornado.web.Application(router)
    app.listen(sys.argv[1])
    tornado.ioloop.IOLoop.current().start()