# -*- coding:utf-8 -*-
from multiprocessing import connection, Pipe
from threading import Thread

import tensorflow as tf
#tf.compat.v1.enable_eager_execution()

import os
import numpy as np
import shutil

from cchess_alphazero.config import Config
from cchess_alphazero.lib.model_helper import load_best_model_weight, need_to_reload_best_model_weight
from cchess_alphazero.lib.web_helper import http_request, download_file
from time import time
from logging import getLogger

logger = getLogger(__name__)

class CChessModelAPI:

    def __init__(self, config: Config, agent_model):  
        self.agent_model = agent_model  # CChessModel
        self.pipes = []     # use for communication between processes/threads
        self.config = config
        self.need_reload = True
        self.done = False

    def start(self, need_reload=True):
        self.need_reload = need_reload
        prediction_worker = Thread(target=self.predict_batch_worker, name="prediction_worker")
        prediction_worker.daemon = True
        prediction_worker.start()

    def get_pipe(self, need_reload=True):
        me, you = Pipe()
        self.pipes.append(me)
        self.need_reload = need_reload
        return you

    #
    def predict_batch_worker(self):
        if self.config.internet.distributed and self.need_reload:
            self.try_reload_model_from_internet()
        last_model_check_time = time()
        while not self.done:
            if last_model_check_time + 600 < time() and self.need_reload:
                self.try_reload_model()
                last_model_check_time = time()
            ready = connection.wait(self.pipes, timeout=0.001)
            if not ready:
                continue
            data, result_pipes, data_len = [], [], []
            for pipe in ready:
                while pipe.poll():
                    try:
                        tmp = pipe.recv()
                    except EOFError as e:
                        logger.error(f"EOF error: {e}")
                        pipe.close()
                    else:
                        data.extend(tmp)
                        data_len.append(len(tmp))
                        result_pipes.append(pipe)
            if not data:
                continue
            data = np.asarray(data, dtype=np.float32)
            #tf.config.experimental_run_functions_eagerly(True)
            policy_ary, value_ary =self.get_predict_values(data)
            #with self.agent_model.graph.as_default():
            #with tf.Graph().as_default():
                #self.agent_model.model.run_eagerly = True
            #policy_ary, value_ary = self.agent_model.model.predict_on_batch(data)
            buf = []
            k, i = 0, 0
            for p, v in zip(policy_ary, value_ary):
                buf.append((p, float(v)))
                k += 1
                if k >= data_len[i]:
                    result_pipes[i].send(buf)
                    buf = []
                    k = 0
                    i += 1
    #@tf.function
    def get_predict_values(self,data):
        policy_ary, value_ary = self.agent_model.model.predict_on_batch(data)
        return policy_ary, value_ary
        
    def try_reload_model(self, config_file=None):
        if config_file:
            config_path = os.path.join(self.config.resource.model_dir, config_file)
            shutil.copy(config_path, self.config.resource.model_best_config_path)
        try:
            if self.config.internet.distributed and not config_file:
                self.try_reload_model_from_internet()
            else:
                if self.need_reload and need_to_reload_best_model_weight(self.agent_model):
                    with self.agent_model.graph.as_default():
                        load_best_model_weight(self.agent_model)
        except Exception as e:
            logger.error(e)

    def try_reload_model_from_internet(self, config_file=None):
        response = http_request(self.config.internet.get_latest_digest)
        if response is None:
            logger.error(f"can't connect to server")
            return
        digest = response['data']['digest']

        if digest != self.agent_model.fetch_digest(self.config.resource.model_best_weight_path):
            logger.info(f"downloading weights please wait...")
            if download_file(self.config.internet.download_url, self.config.resource.model_best_weight_path):
                logger.info(f"finished download weights, start training...")
                try:
                    with self.agent_model.graph.as_default():
                        load_best_model_weight(self.agent_model)
                except ValueError as e:
                    logger.error(f"weight don't match {e}")
                    self.try_reload_model(config_file='model_192x10_config.json')
                except Exception as e:
                    logger.error(f"load weight error: {e},please download again")
                    os.remove(self.config.resource.model_best_weight_path)
                    self.try_reload_model_from_internet()
            else:
                logger.error(f"weight download error")
        else:
            logger.info(f"weight not updated")

    def close(self):
        self.done = True
