#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# File: train-atari.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import time
import numpy as np
import tensorflow as tf
import os, sys, re, time
import random
import uuid
import argparse
import multiprocessing, threading
from collections import deque
import six
from six.moves import queue
from tensorflow.python.framework import ops

from tensorpack import *
from tensorpack.RL.common import MapPlayerState
from tensorpack.RL.gymenv import GymEnv
from tensorpack.RL.history import HistoryFramePlayer
from tensorpack.RL.simulator import SimulatorMaster, SimulatorProcess
from tensorpack.callbacks.stat import StatPrinter
from tensorpack.dataflow.common import BatchData
from tensorpack.dataflow.raw import DataFromQueue
from tensorpack.models.conv2d import Conv2D
from tensorpack.models.model_desc import get_current_tower_context
from tensorpack.models.pool import MaxPooling
from tensorpack.predict.common import PredictConfig
from tensorpack.predict.concurrency import MultiThreadAsyncPredictor
from tensorpack.tfutils.common import get_default_sess_config
from tensorpack.tfutils.gradproc import SummaryGradient
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.train.config import TrainConfig
from tensorpack.train.multigpu import AsyncMultiGPUTrainer, MultiGPUTrainer, SyncMultiGPUTrainer
from tensorpack.utils.concurrency import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.serialize import *
from tensorpack.utils.timer import *
from tensorpack.utils.stat import  *
from tensorpack.tfutils import symbolic_functions as symbf

from tensorpack.RL import *
import common
from common import (play_model, Evaluator, eval_model_multithread)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
parser.add_argument('--load', help='load model')
parser.add_argument('--nr_towers', type=int, default=1)
parser.add_argument('--train_log_path', type=str, default='train_log')
parser.add_argument('--nr_predict_towers', type=int, default=1)
parser.add_argument('--intra_op_par', type=int, default=None)
parser.add_argument('--inter_op_par', type=int, default=None)
parser.add_argument('--cpu_device_count', type=int, default=None)
parser.add_argument('--predict_batch_size', type=int, default=16)
parser.add_argument('--max_steps', type=int, default=None)
parser.add_argument('--env', help='env', required=True)
parser.add_argument('--threads_to_trace', type=int, default=0)
parser.add_argument('--do_train', type=int, default=1)
parser.add_argument('--simulator_procs', type=int, default=20)
parser.add_argument('--task', help='task to perform',
        choices=['play', 'eval', 'train'], default='train')

parser.add_argument('--mkl', help='use the Intel MKL 2D convolution', default=False, type=int)
parser.add_argument('--dummy', type=int, default=0)
parser.add_argument('--dummy_predictor', type=int, default=0)
parser.add_argument('--sync', help='use the synchronous version of the algorithm', type=int, default=0)
parser.add_argument('--channels', help='nr of input channels of the game', type=int, default=3)
parser.add_argument('--steps_per_epoch', type=int, default=250)
parser.add_argument('--cpu', type=int, default=0)
parser.add_argument('--artificial_slowdown', type=float, default=0.0)
parser.add_argument('--queue_size', help='the size of QueueInputTrainer queue', type=int, default=50)
parser.add_argument('--batch_size', help='batch size', type=int, default=128)
parser.add_argument('--my_sim_master_queue', help='MySimulatorMaster queue size', type=int, default=2048)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--data_source_delay', type=int, help='artificial delay in the pipeline, in batches', default=0)
parser.add_argument('--max_epoch', type=int, help='number of epochs after which to stop', default=1000)
parser.add_argument('--epochs_for_evaluation', type=int, help='number of epochs to pass between evalations', default=2)
args = parser.parse_args()

logger.info(str(args))

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
GAMMA = 0.99
CHANNEL = FRAME_HISTORY * args.channels
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

LOCAL_TIME_MAX = 5
STEP_PER_EPOCH = args.steps_per_epoch
EVAL_EPISODE = 50

BATCH_SIZE = args.batch_size

#PREDICTOR_THREAD_PER_GPU = 8
#PREDICTOR_THREAD = 1
#EVALUATE_PROC = min(multiprocessing.cpu_count() // 2, 20)

NUM_ACTIONS = None
ENV_NAME = None


class DelayingDataSource(ProxyDataFlow):
    """
    This class is designed to test if introducing a delay
    of several training batches can have negative effects
    on the convergence of an asynchronous reinforcement
    learning algorithm such as BA3C.
    
    The class introduces a delay of configurable length
    into the data pipeline. The hypothesis is that with
    a big enough delay the algorithm will stop learning
    anything meaningful.    
    """
    def __init__(self, ds, delay):
        super(DelayingDataSource, self).__init__(ds)
        self.delay = delay
        self.buffer = []
        
    def size(self):
        ds_size = self.ds.size()
        return ds_size
    
    def get_data(self):
        while True:
            assert len(self.buffer) <= self.delay
            while len(self.buffer) != self.delay:
                new_dp = next(self.ds.get_data())
                self.buffer.append(new_dp)

            new_dp = next(self.ds.get_data())
            self.buffer.append(new_dp)
            oldest_dp = self.buffer.pop(0)
            yield oldest_dp


##### TF SLEEP #####
# Define custom py_func which takes also a grad op as argument:
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    
    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

# Def custom square function using np.square instead of tf.square:
def tf_sleep(x, name=None, sleep=1):
    def f(x):
        #print('sleeping for: ', sleep)
        time.sleep(sleep)
        return x

    with ops.op_scope([x], name, "tf_sleep") as name:
        sqr_x = py_func(f,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=_tf_sleep_grad)  # <-- here's the call to the gradient
        return sqr_x[0]

def _tf_sleep_grad(op, grad):
    x = op.inputs[0]
    #print(x)
    #print(grad)
    return grad #grad # * 20 * x  # add a "small" error just to see the difference:
##### TF SLEEP END #####


def get_player(viz=False, train=False, dumpdir=None):
    pl = GymEnv(ENV_NAME, dumpdir=dumpdir)
    def func(img):
        return cv2.resize(img, IMAGE_SIZE[::-1])
    pl = MapPlayerState(pl, func)

    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_action_space().num_actions()

    pl = HistoryFramePlayer(pl, FRAME_HISTORY)
    if not train:
        pl = PreventStuckPlayer(pl, 30, 1)
    pl = LimitLengthPlayer(pl, 40000)
    return pl
common.get_player = get_player

class MySimulatorWorker(SimulatorProcess):
    def _build_player(self):
        return get_player(train=True)

class Model(ModelDesc):
    def _get_input_vars(self):
        assert NUM_ACTIONS is not None
        return [InputVar(tf.float32, (None,) + IMAGE_SHAPE3, 'state'),
                InputVar(tf.int64, (None,), 'action'),
                InputVar(tf.float32, (None,), 'futurereward') ]

    def _get_NN_prediction(self, image):
        image = image / 255.0
        
        print(CHANNEL)
        dummy_channels = 16 - CHANNEL
        image = tf.concat(3, [image, tf.zeros((tf.shape(image)[0], 84, 84, dummy_channels))])
        with argscope(Conv2D, nl=tf.nn.relu):
            input_sum = tf.reduce_sum(tf.abs(image))
            #i = tf.Print(image, [image], message='input image: ', summarize=30)
            #i2 = tf.Print(i, [input_sum], message='input abs sum: ')            
            l = Conv2D('conv0', image, out_channel=32, kernel_shape=5, use_bias=False, padding='VALID')
            #l = tf.Print(l, [l], message='conv0: ', summarize=30)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, out_channel=32, kernel_shape=5, use_bias=False, padding='VALID')
            #l = tf.Print(l, [l], message='conv1: ', summarize=30)            
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, out_channel=64, kernel_shape=5, use_bias=False, padding='VALID')

            #l = tf.Print(l, [l], message='conv2: ', summarize=30)                        
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, out_channel=64, kernel_shape=3, use_bias=False, padding='VALID')

        #l = tf.Print(l, [l], message='conv3: ', summarize=30)
        l = FullyConnected('fc0', l, 512, nl=tf.identity)
        if args.artificial_slowdown != 0.0:
            l2 = tf_sleep(l, sleep=args.artificial_slowdown)
            l = tf.reshape(l2, tf.shape(l))#.get_shape())
        #l = tf.Print(l, [l], message='fc0: ', summarize=15)
        
        l = PReLU('prelu', l)
        policy = FullyConnected('fc-pi', l, out_dim=NUM_ACTIONS, nl=tf.identity)
        value = FullyConnected('fc-v', l, 1, nl=tf.identity)
        return policy, value

    def _build_graph(self, inputs):
        #import ipdb; ipdb.set_trace()
        state, action, futurereward = inputs
        policy, self.value = self._get_NN_prediction(state)
        self.value = tf.squeeze(self.value, [1], name='pred_value') # (B,)
        self.logits = tf.nn.softmax(policy, name='logits')

        expf = tf.get_variable('explore_factor', shape=[],
                initializer=tf.constant_initializer(1), trainable=False)
        logitsT = tf.nn.softmax(policy * expf, name='logitsT')
        is_training = get_current_tower_context().is_training
        if not is_training:
            return
        log_probs = tf.log(self.logits + 1e-6)

        log_pi_a_given_s = tf.reduce_sum(
                log_probs * tf.one_hot(action, NUM_ACTIONS), 1)
        advantage = tf.sub(tf.stop_gradient(self.value), futurereward, name='advantage')
        policy_loss = tf.reduce_sum(log_pi_a_given_s * advantage, name='policy_loss')
        policy_loss = tf.Print(policy_loss, [policy_loss], 'policy_loss')
        
        xentropy_loss = tf.reduce_sum(
                self.logits * log_probs, name='xentropy_loss')
        value_loss = tf.nn.l2_loss(self.value - futurereward, name='value_loss')
        value_loss = tf.Print(value_loss, [value_loss], 'value loss: ')

        pred_reward = tf.reduce_mean(self.value, name='predict_reward')
        pred_reward = tf.Print(pred_reward, [pred_reward], 'pred_reward: ')
        
        advantage = symbf.rms(advantage, name='rms_advantage')
        summary.add_moving_summary(policy_loss, xentropy_loss, value_loss, pred_reward, advantage)
        entropy_beta = tf.get_variable('entropy_beta', shape=[],
                initializer=tf.constant_initializer(0.01), trainable=False)
        self.cost = tf.add_n([policy_loss, xentropy_loss * entropy_beta, value_loss])
        self.cost = tf.truediv(self.cost,
                tf.cast(tf.shape(futurereward)[0], tf.float32),
                name='cost')
        self.cost = tf.Print(self.cost, [self.cost], 'cost: ')

    def get_gradient_processor(self):
        return [MapGradient(lambda grad: tf.clip_by_average_norm(grad, 0.1)),
                SummaryGradient()]

class MySimulatorMaster(SimulatorMaster, Callback):
    def __init__(self, pipe_c2s, pipe_s2c, model, dummy, predictor_threads, predict_batch_size=16, do_train=True):
        # predictor_threads is previous PREDICTOR_THREAD
        super(MySimulatorMaster, self).__init__(pipe_c2s, pipe_s2c)
        self.M = model
        self.do_train = do_train
        
        #the second queue is here!
        self.queue = queue.Queue(maxsize=args.my_sim_master_queue)
        self.dummy = dummy
        self.predictor_threads = predictor_threads

        self.last_queue_put = 0
        self.queue_put_times = []
        self.predict_batch_size = predict_batch_size
        self.counter = 0

    def _setup_graph(self):
        self.sess = self.trainer.sess
        self.async_predictor = MultiThreadAsyncPredictor(
                self.trainer.get_predict_funcs(['state'], ['logitsT', 'pred_value'], self.predictor_threads),
            batch_size=self.predict_batch_size)
        self.async_predictor.run()

    def _on_state(self, state, ident):
        ident, ts = ident
        client = self.clients[ident]


        if self.dummy:
            action = 0
            value = 0.0
            client.memory.append(TransitionExperience(state, action, None, value=value))
            self.send_queue.put([ident, dumps(action)])
        else:
            def cb(outputs):
                distrib, value = outputs.result()
                #distrib, value, ts = outputs.result()
                
                #print '_on_state cb', distrib
                assert np.all(np.isfinite(distrib)), distrib
                action = np.random.choice(len(distrib), p=distrib)
                client = self.clients[ident]
                client.memory.append(TransitionExperience(state, action, None, value=value))
                #print("Q-debug: MySimulatorMaster send_queue before put, size: ", self.send_queue.qsize(), '/', self.send_queue.maxsize)
                ts = 0.0
                self.send_queue.put([ident, dumps((action, ts))])
            self.async_predictor.put_task([state], cb)

    def _on_episode_over(self, ident):
        ident, ts = ident
        self._parse_memory(0, ident, True, ts)

    def _on_datapoint(self, ident):
        ident, ts = ident
        client = self.clients[ident]
        if len(client.memory) == LOCAL_TIME_MAX + 1:
            R = client.memory[-1].value
            self._parse_memory(R, ident, False, ts)

    def _parse_memory(self, init_r, ident, isOver, ts):
        client = self.clients[ident]
        mem = client.memory
        if not isOver:
            last = mem[-1]
            mem = mem[:-1]

        mem.reverse()
        R = float(init_r)
        for idx, k in enumerate(mem):
            #print '### reward: ', k.reward
            R = np.clip(k.reward, -1, 1) + GAMMA * R
            #print("Q-debug id=39dksc: MySimulatorMaster self.queue before put, size: ", self.queue.qsize(), '/', self.queue.maxsize)
            logger.debug("Q-debug id=39dksc: MySimulatorMaster self.queue before put, size: {qsize} / {maxsize}".format(
                qsize=self.queue.qsize(),
                maxsize=self.queue.maxsize))
            self.log_queue_put()
            if self.do_train:
                self.queue.put([k.state, k.action, R, ts])

        if not isOver:
            client.memory = [last]
        else:
            client.memory = []

    def log_queue_put(self):
        self.counter += 1
        elapsed_last_put = 0
        self.queue_put_times.append(elapsed_last_put)
        k = 1000
        if self.counter % 1 == 0:
            logger.debug("queue_put_times elapsed {elapsed}".format(elapsed=elapsed_last_put))
            logger.debug("queue_put_times {puts_s} puts/s".format(puts_s=1000.0 / np.mean(self.queue_put_times[-k:])))
        self.last_queue_put = 0


def get_config(args=None):
    logger.set_logger_dir(args.train_log_path)
    #logger.auto_set_dir()
    M = Model()

    name_base = str(uuid.uuid1())[:6]
    PIPE_DIR = os.environ.get('TENSORPACK_PIPEDIR', '.').rstrip('/')
    namec2s = 'ipc://{}/sim-c2s-{}'.format(PIPE_DIR, name_base)
    names2c = 'ipc://{}/sim-s2c-{}'.format(PIPE_DIR, name_base)
    procs = [MySimulatorWorker(k, namec2s, names2c) for k in range(args.simulator_procs)]
    ensure_proc_terminate(procs)
    start_proc_mask_signal(procs)

    master = MySimulatorMaster(namec2s, names2c, M, dummy=args.dummy,
                               predictor_threads=args.nr_predict_towers, predict_batch_size=args.predict_batch_size,
                               do_train=args.do_train)
    
    #here's the data passed to the repeated data source
    dataflow = BatchData(DataFromQueue(master.queue), BATCH_SIZE)
    dataflow = DelayingDataSource(dataflow, args.data_source_delay)

    lr = tf.Variable(args.learning_rate, trainable=False, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    intra_op_par = args.intra_op_par
    inter_op_par = args.inter_op_par

    session_config = get_default_sess_config(0.5)
    if intra_op_par is not None:
        session_config.intra_op_parallelism_threads = intra_op_par

    if inter_op_par is not None:
        session_config.inter_op_parallelism_threads = inter_op_par

    session_config.log_device_placement = False
    extra_arg = {
        'dummy_predictor': args.dummy_predictor,
        'intra_op_par': intra_op_par,
        'inter_op_par': inter_op_par,
        'max_steps': args.max_steps,
        'device_count': {'CPU': args.cpu_device_count},
        'threads_to_trace': args.threads_to_trace,
        'dummy': args.dummy,
        'cpu' : args.cpu,
        'queue_size' : args.queue_size
    }

    return TrainConfig(
        dataset=dataflow,
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
        callbacks=Callbacks([
            StatPrinter(), ModelSaver(),

            ScheduledHyperParamSetter('learning_rate', [(80, 0.0003), (120, 0.0001)]),
            ScheduledHyperParamSetter('entropy_beta', [(80, 0.005)]),
            ScheduledHyperParamSetter('explore_factor',
                [(80, 2), (100, 3), (120, 4), (140, 5)]),

            HumanHyperParamSetter('learning_rate'),
            HumanHyperParamSetter('entropy_beta'),
            HumanHyperParamSetter('explore_factor'),
            master,
            PeriodicCallback(Evaluator(EVAL_EPISODE, ['state'], ['logits']), args.epochs_for_evaluation),
        ]),
        extra_threads_procs=[master],
        session_config=session_config,
        model=M,
        step_per_epoch=STEP_PER_EPOCH,
        max_epoch=args.max_epoch,
        extra_arg=extra_arg
    )

if __name__ == '__main__':
    import os
    os.setpgrp()

    tf.logging.set_verbosity(tf.logging.ERROR)
    import logging
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    

    print "args.mkl == ", args.mkl
    ENV_NAME = args.env
    p = get_player(); del p    # set NUM_ACTIONS

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.task != 'train':
        assert args.load is not None

    if args.mkl:
        print "using MKL convolution"
        label_map = {"Conv2D": "MKL",
                     "Conv2DBackpropFilter": "MKL",
                     "Conv2DBackpropInput": "MKL"}
    else:
        print "using tensorflow convolution"
        label_map = {}


    with ops.Graph().as_default() as g:
        tf.set_random_seed(0)
        with g._kernel_label_map(label_map):
            if args.task != 'train':
                cfg = PredictConfig(
                        model=Model(),
                        session_init=SaverRestore(args.load),
                        input_var_names=['state'],
                        output_var_names=['logits:0'])
                if args.task == 'play':
                    play_model(cfg)
                elif args.task == 'eval':
                    eval_model_multithread(cfg, EVAL_EPISODE)
            else:
                nr_towers = args.nr_towers
                predict_towers = args.nr_predict_towers * [0]
                
                if args.cpu != 1:
                    nr_gpu = get_nr_gpu()
                    if nr_gpu > 1:
                        predict_tower = range(nr_gpu)[-nr_gpu/2:]
                    else:
                        predict_tower = [0]
    
                    #PREDICTOR_THREAD = len(predict_tower) * PREDICTOR_THREAD_PER_GPU
                #PREDICTOR_THREAD = 10


                config = get_config(args)
                if args.load:
                    config.session_init = SaverRestore(args.load)
                config.tower = range(nr_towers)

                logger.info("[BA3C] Train on gpu {} and infer on gpu {}".format(
                    ','.join(map(str, config.tower)), ','.join(map(str, predict_towers))))


                if args.sync:
                    logger.info('using sync version')
                    SyncMultiGPUTrainer(config, predict_tower=predict_towers).train()
                else:
                    logger.info('using async version')
                    AsyncMultiGPUTrainer(config, predict_tower=predict_towers).train()
