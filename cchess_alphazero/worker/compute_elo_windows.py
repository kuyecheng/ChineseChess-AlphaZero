import os
import gc
import sys
import hashlib
import numpy as np
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from logging import getLogger
from multiprocessing import Manager
from threading import Thread
from time import time
from collections import defaultdict
from threading import Lock
from time import sleep
from random import random, randint

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.agent.player import CChessPlayer, VisitState
from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner, ActionLabelsRed, flip_policy, flip_move
from cchess_alphazero.lib.data_helper import write_game_data_to_file
from cchess_alphazero.lib.model_helper import load_model_weight
from cchess_alphazero.lib.web_helper import upload_file, download_file, http_request

logger = getLogger(__name__)

job_done = Lock()
thr_free = Lock()
rst = None
data = None
futures =[]

def start(config: Config):
    return EvaluateWorker(config).start()

class EvaluateWorker:
    def __init__(self, config: Config):
        self.config = config
        self.m = Manager()
        self.pipes_bt = None
        self.pipes_ng = None
        self.data = None

    def start(self):
        global job_done
        global thr_free
        global rst
        global data
        global futures

        job_done.acquire(True)
        response = http_request(self.config.internet.get_evaluate_model_url)
        if not response or int(response['status']) != 0:
            logger.info(f"no weight under evaluate")
            sys.exit()
        self.data = response['data']
        logger.info(f"start evalute: base model: {self.data['base']['digest'][0:8]}, elo = {self.data['base']['elo']};"
                    f"model under evaluate: {self.data['unchecked']['digest'][0:8]}")
        # weight path
        base_weight_path = os.path.join(self.config.resource.next_generation_model_dir, self.data['base']['digest'] + '.h5')
        ng_weight_path = os.path.join(self.config.resource.next_generation_model_dir, self.data['unchecked']['digest'] + '.h5')
        # load model
        model_base, hist_base = self.load_model(base_weight_path, self.data['base']['digest'])
        model_ng, hist_ng = self.load_model(ng_weight_path, self.data['unchecked']['digest'])
        # make pipes
        self.pipes_bt = self.m.list([model_base.get_pipes(need_reload=False) for _ in range(self.config.play.max_processes)])
        self.pipes_ng = self.m.list([model_ng.get_pipes(need_reload=False) for _ in range(self.config.play.max_processes)])

        need_evaluate = True
        self.config.opts.evaluate = True
        
        with ProcessPoolExecutor(max_workers=self.config.play.max_processes) as executor:
            while need_evaluate:
                start_time = time()

                if len(futures) == 0:
                    for i in range(self.config.play.max_processes):
                        idx = 0 if random() > 0.5 else 1
                        ff = executor.submit(self_play_buffer, self.config, self.pipes_bt, self.pipes_ng, 
                                            idx, self.data, hist_base, hist_ng)
                        ff.add_done_callback(recall_fn)
                        futures.append(ff)

                job_done.acquire(True)

                end_time = time()

                turns = rst[0]
                value = rst[1]
                idx = rst[2]

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

                logger.info(f"finished, time{(end_time - start_time):.1f} seconds "
                             f"{turns / 2}round, {result}, score: {score}, value = {value}, idx = {idx}")

                response = self.save_play_data(idx, data, value, score)
                if response and int(response['status']) == 0:
                    logger.info('upload done ')
                else:
                    logger.info(f"upload failed. return {response}")

                response = http_request(self.config.internet.get_evaluate_model_url)
                if int(response['status']) == 0 and response['data']['base']['digest'] == self.data['base']['digest']\
                    and response['data']['unchecked']['digest'] == self.data['unchecked']['digest']:
                    need_evaluate = True
                    logger.info(f"continue evaluate")
                    idx = 0 if random() > 0.5 else 1
                    ff = executor.submit(self_play_buffer, self.config, self.pipes_bt, self.pipes_ng, 
                                        idx, self.data, hist_base, hist_ng)
                    ff.add_done_callback(recall_fn)
                    futures.append(ff) # Keep it going
                else:
                    need_evaluate = False
                    logger.info(f"stop evaluate")
                thr_free.release()

        model_base.close_pipes()
        model_ng.close_pipes()

    def load_model(self, weight_path, digest, config_file=None):
        model = CChessModel(self.config)
        use_history = False
        if not config_file:
            use_history = False
            config_path = self.config.resource.model_best_config_path
        else:
            config_path = os.path.join(self.config.resource.model_dir, config_file)
        if (not load_model_weight(model, config_path, weight_path)) or model.digest != digest:
            logger.info(f"start download weight {digest[0:8]}")
            url = self.config.internet.download_base_url + digest + '.h5'
            download_file(url, weight_path)
            try:
                if not load_model_weight(model, config_path, weight_path):
                    logger.info(f"weight not uploaded")
                    sys.exit()
            except ValueError as e:
                logger.error(f"weight don't match {e}")
                return self.load_model(weight_path, digest, 'model_192x10_config.json')
            except Exception as e:
                logger.error(f"weight load error: {e}  later download again")
                os.remove(weight_path)
                sleep(10)
                return self.load_model(weight_path, digest)
        logger.info(f"load weight {model.digest[0:8]} success")
        return model, use_history

    def save_play_data(self, idx, data, value, score):
        rc = self.config.resource
        game_id = datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        filename = rc.play_data_filename_tmpl % game_id
        path = os.path.join(rc.play_data_dir, filename)
        logger.info("save data to %s" % (path))
        write_game_data_to_file(path, data)
        logger.info(f"upload game to {filename} ...")
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

def recall_fn(future):
    global thr_free
    global job_done
    global rst
    global data
    global futures

    thr_free.acquire(True)
    rst, data = future.result()
    futures.remove(future)
    job_done.release()

def self_play_buffer(config, pipes_bt, pipes_ng, idx, res_data, hist_base, hist_ng) -> (tuple, list):
    sleep(random())
    playouts = randint(8, 12) * 100
    config.play.simulation_num_per_move = playouts
    logger.info(f"Set playouts = {config.play.simulation_num_per_move}")

    pipe1 = pipes_bt.pop() # borrow
    pipe2 = pipes_ng.pop()

    player1 = CChessPlayer(config, search_tree=defaultdict(VisitState), pipes=pipe1, 
        enable_resign=False, debugging=False, use_history=hist_base)
    player2 = CChessPlayer(config, search_tree=defaultdict(VisitState), pipes=pipe2, 
        enable_resign=False, debugging=False, use_history=hist_ng)

    # even: bst = red, ng = black; odd: bst = black, ng = red
    if idx % 2 == 0:
        red = player1
        black = player2
        print(f"base model red")
    else:
        red = player2
        black = player1
        print(f"base model black")

    state = senv.INIT_STATE
    history = [state]
    # policys = [] 
    value = 0
    turns = 0
    game_over = False
    final_move = None
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
        if action is None:
            print(f"{turns % 2} (0 = red, 1 = black) loss")
            value = -1
            break
        print(f"playing: round{turns / 2 + 1} {'red move' if turns % 2 == 0 else 'black move'}, method: {action}, time: {(end_time - start_time):.1f}s")
        # policys.append(policy)
        history.append(action)
        try:
            state, no_eat = senv.new_step(state, action)
        except Exception as e:
            logger.error(f"{e}, no_act = {no_act}, policy = {policy}")
            game_over = True
            value = 0
            break
        turns += 1
        if no_eat:
            no_eat_count += 1
        else:
            no_eat_count = 0
        history.append(state)

        if no_eat_count >= 120 or turns / 2 >= config.play.max_game_length:
            game_over = True
            value = 0
        else:
            game_over, value, final_move, check = senv.done(state, need_check=True)
            no_act = []
            increase_temp = False
            if not game_over:
                if not senv.has_attack_chessman(state):
                    logger.info(f"draw. no soldier, state = {state}")
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
                                logger.info("draw with 3 ")
                                break

    if final_move:
        history.append(final_move)
        state = senv.step(state, final_move)
        turns += 1
        value = -value
        history.append(state)

    data = []
    if idx % 2 == 0:
        data = [res_data['base']['digest'], res_data['unchecked']['digest']]
    else:
        data = [res_data['unchecked']['digest'], res_data['base']['digest']]
    player1.close()
    player2.close()
    del player1, player2
    gc.collect()

    if turns % 2 == 1:  # balck turn
        value = -value
    
    v = value
    data.append(history[0])
    for i in range(turns):
        k = i * 2
        data.append([history[k + 1], value])
        value = -value

    pipes_bt.append(pipe1)
    pipes_ng.append(pipe2)
    return (turns, v, idx), data

def build_policy(action, flip):
    labels_n = len(ActionLabelsRed)
    move_lookup = {move: i for move, i in zip(ActionLabelsRed, range(labels_n))}
    policy = np.zeros(labels_n)

    policy[move_lookup[action]] = 1

    if flip:
        policy = flip_policy(policy)
    return list(policy)
