import os
import sys
import shutil
import hashlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait
from datetime import datetime
from logging import getLogger
from multiprocessing import Manager
from threading import Thread
from time import time, sleep
from collections import defaultdict
from multiprocessing import Lock
from random import random, randint
import numpy as np
import subprocess

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.agent.player import CChessPlayer, VisitState
from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner, flip_move, ActionLabelsRed
from cchess_alphazero.lib.data_helper import get_game_data_filenames, write_game_data_to_file
from cchess_alphazero.lib.model_helper import load_model_weight
from cchess_alphazero.lib.tf_util import set_session_config
from cchess_alphazero.lib.web_helper import http_request, download_file
from cchess_alphazero.lib.elo_helper import compute_elo
from cchess_alphazero.lib.web_helper import upload_file

logger = getLogger(__name__)

def start(config: Config):
    set_session_config(per_process_gpu_memory_fraction=1, allow_growth=True, device_list=config.opts.device_list)
    m = Manager()
    response = http_request(config.internet.get_evaluate_model_url)
    while int(response['status']) == 0:
        data = response['data']
        logger.info(f"start evaluate: base model {data['base']['digest'][0:8]}, elo = {data['base']['elo']};"
                    f"model under evalute: {data['unchecked']['digest'][0:8]}, elo = {data['unchecked']['elo']}")
        # make path
        base_weight_path = os.path.join(config.resource.next_generation_model_dir, data['base']['digest'] + '.h5')
        ng_weight_path = os.path.join(config.resource.next_generation_model_dir, data['unchecked']['digest'] + '.h5')
        # load model
        model_base, hist_base = load_model(config, base_weight_path, data['base']['digest'])
        model_ng, hist_ng = load_model(config, ng_weight_path, data['unchecked']['digest'])
        # make pipes
        model_base_pipes = m.list([model_base.get_pipes(need_reload=False) for _ in range(config.play.max_processes)])
        model_ng_pipes = m.list([model_ng.get_pipes(need_reload=False) for _ in range(config.play.max_processes)])

        # eval_worker = EvaluateWorker(config, model_base_pipes, model_ng_pipes, 0, data)
        # res = eval_worker.start()
        with ProcessPoolExecutor(max_workers=config.play.max_processes) as executor:
            futures = []
            for i in range(config.play.max_processes):
                eval_worker = EvaluateWorker(config, model_base_pipes, model_ng_pipes, i, data, hist_base, hist_ng)
                futures.append(executor.submit(eval_worker.start))
                sleep(1)
        
        wait(futures)
        # close pipe
        model_base.close_pipes()
        model_ng.close_pipes()
        # reset model
        model_base = None
        model_ng = None

        response = http_request(config.internet.get_evaluate_model_url)
    logger.info(f"no weight under evaluate")

class EvaluateWorker:
    def __init__(self, config: Config, pipes1=None, pipes2=None, pid=None, data=None, hist_base=True, hist_ng=True):
        self.config = config
        self.player_bt = None
        self.player_ng = None
        self.pid = pid
        self.pipes_bt = pipes1
        self.pipes_ng = pipes2
        self.data = data
        self.hist_base = hist_base
        self.hist_ng = hist_ng

    def start(self):
        sleep((self.pid % self.config.play.max_processes) * 10)
        logger.debug(f"Evaluate#Start Process index = {self.pid}, pid = {os.getpid()}")
        need_evaluate = True
        self.config.opts.evaluate = True

        while need_evaluate:
            idx = 0 if random() > 0.5 else 1
            start_time = time()
            value, turns, data = self.start_game(idx)
            end_time = time()
            
            if (value == 1 and idx == 0) or (value == -1 and idx == 1):
                result = 'base model win'
            elif (value == 1 and idx == 1) or (value == -1 and idx == 0):
                result = 'under evaluate win'
            else:
                result = 'draw'

            if value == -1: # loss
                score = 0
            elif value == 1: # win
                score = 1
            else:
                score = 0.5

            if idx == 0:
                score = 1 - score
            else:
                score = score

            logger.info(f"process {self.pid}finished,time {(end_time - start_time):.1f} seconds,"
                         f"{turns / 2}rounds, {result}, score: {score}, value = {value}, idx = {idx}")

            response = self.save_play_data(idx, data, value, score)
            if response and int(response['status']) == 0:
                logger.info('upload success')
            else:
                logger.info(f"upload failed, return {response}")

            response = http_request(self.config.internet.get_evaluate_model_url)
            if int(response['status']) == 0 and response['data']['base']['digest'] == self.data['base']['digest']\
                and response['data']['unchecked']['digest'] == self.data['unchecked']['digest']:
                need_evaluate = True
                logger.info(f"process{self.pid} continue evaluate")
            else:
                need_evaluate = False
                logger.info(f"process {self.pid} stop evaluate")


    def start_game(self, idx):
        sleep(random())
        playouts = randint(8, 12) * 100
        self.config.play.simulation_num_per_move = playouts
        logger.info(f"Set playouts = {self.config.play.simulation_num_per_move}")

        pipe1 = self.pipes_bt.pop()
        pipe2 = self.pipes_ng.pop()
        search_tree1 = defaultdict(VisitState)
        search_tree2 = defaultdict(VisitState)

        self.player1 = CChessPlayer(self.config, search_tree=search_tree1, pipes=pipe1, 
                        debugging=False, enable_resign=False, use_history=self.hist_base)
        self.player2 = CChessPlayer(self.config, search_tree=search_tree2, pipes=pipe2, 
                        debugging=False, enable_resign=False, use_history=self.hist_ng)

        # even: bst = red, ng = black; odd: bst = black, ng = red
        if idx % 2 == 0:
            red = self.player1
            black = self.player2
            logger.info(f"process id = {self.pid} base model red")
        else:
            red = self.player2
            black = self.player1
            logger.info(f"process id = {self.pid} base model black")

        state = senv.INIT_STATE
        history = [state]
        value = 0       # best model's value
        turns = 0       # even == red; odd == black
        game_over = False
        no_eat_count = 0
        check = False
        increase_temp = False
        no_act = []

        while not game_over:
            start_time = time()
            if turns % 2 == 0:
                action, _ = red.action(state, turns, no_act=no_act, increase_temp=increase_temp)
            else:
                action, _ = black.action(state, turns, no_act=no_act, increase_temp=increase_temp)
            end_time = time()
            if self.config.opts.log_move:
                logger.debug(f"process id = {self.pid}, action = {action}, turns = {turns}, time = {(end_time-start_time):.1f}")
            if action is None:
                logger.debug(f"{turns % 2} (0 = red; 1 = black) has resigned")
                value = -1
                break
            history.append(action)
            state, no_eat = senv.new_step(state, action)
            turns += 1
            if no_eat:
                no_eat_count += 1
            else:
                no_eat_count = 0
            history.append(state)

            if no_eat_count >= 120 or turns / 2 >= self.config.play.max_game_length:
                game_over = True
                value = 0
            else:
                game_over, value, final_move, check = senv.done(state, need_check=True)
                no_act = []
                increase_temp = False
                if not game_over:
                    if not senv.has_attack_chessman(state):
                        logger.info(f"no fighting soldier, draw state = {state}")
                        game_over = True
                        value = 0
                if not game_over and not check and state in history[:-1]:
                    free_move = defaultdict(int)
                    for i in range(len(history) - 1):
                        if history[i] == state:
                            if senv.will_check_or_catch(state, history[i+1]):
                                no_act.append(history[i + 1])
                            elif not senv.be_catched(state, history[i+1]):
                                increase_temp = True
                                free_move[state] += 1
                                if free_move[state] >= 3:
                                   
                                    game_over = True
                                    value = 0
                                    logger.info("draw with 3 moves")
                                    break

        if final_move:
            history.append(final_move)
            state = senv.step(state, final_move)
            turns += 1
            value = - value
            history.append(state)

        data = []
        if idx % 2 == 0:
            data = [self.data['base']['digest'], self.data['unchecked']['digest']]
        else:
            data = [self.data['unchecked']['digest'], self.data['base']['digest']]
        self.player1.close()
        self.player2.close()

        if turns % 2 == 1:  # black turn
            value = -value

        v = value
        data.append(history[0])
        for i in range(turns):
            k = i * 2
            data.append([history[k + 1], v])
            v = -v

        self.pipes_bt.append(pipe1)
        self.pipes_ng.append(pipe2)
        return value, turns, data

    def save_play_data(self, idx, data, value, score):
        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        filename = rc.play_data_filename_tmpl % game_id
        path = os.path.join(rc.play_data_dir, filename)
        logger.info(f"Process {self.pid} save play data to {path}")
        write_game_data_to_file(path, data)
        logger.info(f"Uploading play data {filename} ...")
        red, black = data[0], data[1]
        return self.upload_eval_data(path, filename, red, black, value, score)

    def upload_eval_data(self, path, filename, red, black, result, score):
        hash = self.fetch_digest(path)
        data = {'digest': self.data['unchecked']['digest'], 'red_digest': red, 'black_digest': black, 
                'result': result, 'score': score, 'hash': hash}
        response = upload_file(self.config.internet.upload_eval_url, path, filename, data, rm=False)
        return response

    def fetch_digest(self, file_path):
        if os.path.exists(file_path):
            m = hashlib.sha256()
            with open(file_path, "rb") as f:
                m.update(f.read())
            return m.hexdigest()
        return None


def load_model(config, weight_path, digest, config_file=None):
    model = CChessModel(config)
    use_history = False
    if not config_file:
        config_path = config.resource.model_best_config_path
        use_history = False
    else:
        config_path = os.path.join(config.resource.model_dir, config_file)
    logger.debug(f"config_path = {config_path}, digest = {digest}")
    if (not load_model_weight(model, config_path, weight_path)) or model.digest != digest:
        logger.info(f"start download weight {digest[0:8]}")
        url = config.internet.download_base_url + digest + '.h5'
        download_file(url, weight_path)
        try:
            if not load_model_weight(model, config_path, weight_path):
                logger.info(f"weight not uploaded")
                sys.exit()
        except ValueError as e:
            logger.error(f"weight doesn't match {e}")
            return load_model(config, weight_path, digest, 'model_192x10_config.json')
        except Exception as e:
            logger.error(f"load weight error {e}1 0s download again")
            os.remove(weight_path)
            sleep(10)
            return load_model(config, weight_path, digest)
    logger.info(f"load weight {digest[0:8]} success")
    return model, use_history


