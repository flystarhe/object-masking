import json
import traceback
from .base import Base
from tornado.web import HTTPError
from tornado.concurrent import run_on_executor
class Guess(Base):
    @run_on_executor(executor='thread_pool')
    def worker(self):
        arguments = self.request.arguments
        try:
            var = arguments['name'][0].decode('utf8')
            # computing task
            msg = str(var)
            # result
            res = json.dumps({'state': 0, 'result': msg})
            log = 'name:{}'.format(var)
        except Exception:
            raise HTTPError(400, traceback.format_exc())
        return res, log