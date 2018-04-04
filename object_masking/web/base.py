#coding=utf-8
import logging
import logging.handlers as handlers
filehandler = handlers.RotatingFileHandler('log.web', maxBytes=10**6, backupCount=1)
filehandler.setLevel(logging.INFO)
filehandler.setFormatter(logging.Formatter('%(asctime)s-%(levelname)s:%(message)s'))
logger = logging.getLogger('web')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)

import tornado.web
from tornado import gen
from tornado.concurrent import futures
class Base(tornado.web.RequestHandler):
    thread_pool = futures.ThreadPoolExecutor(2)

    def set_default_headers(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Headers', 'Content-Type')
        self.set_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')

    def options(self):
        #raise HTTPError(405)  #若允许跨域请屏蔽
        self.set_status(200)
        self.finish()

    @gen.coroutine
    def get(self):
        res, log = yield self.worker()
        self.write(res)
        self.finish()
        logger.info(log)

    @gen.coroutine
    def post(self):
        res, log = yield self.worker()
        self.write(res)
        self.finish()
        logger.info(log)