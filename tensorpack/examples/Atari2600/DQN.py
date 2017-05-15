#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# File: DQN.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
import tensorflow as tf

import os, sys, re, time
import random
import argparse
import subprocess
import multiprocessing, threading
from collections import deque

from tensorpack import *
from tensorpack.utils.concurrency import *
from tensorpack.tfutils import symbolic_functions as symbf
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.RL import *

import common
from common import play_model, Evaluator, eval_model_multithread
from atari import AtariPlayer

BATCH_SIZE = 64
IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
ACTION_REPEAT = 4
HEIGHT_RANGE = (None, None)
#HEIGHT_RANGE = (36, 204)    # for breakout
#HEIGHT_RANGE = (28, -8)   # for pong

CHANNEL = FRAME_HISTORY
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)
GAMMA = 0.99

INIT_EXPLORATION = 1
EXPLORATION_EPOCH_ANNEAL = 0.01
END_EXPLORATION = 0.1

MEMORY_SIZE = 1e6
INIT_MEMORY_SIZE = 5e4
STEP_PER_EPOCH = 10000
EVAL_EPISODE = 50

NUM_ACTIONS = None
ROM_FILE = None

def get_player(viz=False, train=False):
    pl = AtariPlayer(ROM_FILE, height_range=HEIGHT_RANGE,
            frame_skip=ACTION_REPEAT, image_shape=IMAGE_SIZE[::-1], viz=viz,
            live_lost_as_eoe=train)
    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_action_space().num_actions()
    if not train:
        pl = HistoryFramePlayer(pl, FRAME_HISTORY)
        pl = PreventStuckPlayer(pl, 30, 1)
    pl = LimitLengthPlayer(pl, 30000)
    return pl
common.get_player = get_player  # so that eval functions in common can use the player

class Model(ModelDesc):
    def _get_input_vars(self):
        if NUM_ACTIONS is None:
            p = get_player(); del p
        return [InputVar(tf.float32, (None,) + IMAGE_SHAPE3, 'state'),
                InputVar(tf.int64, (None,), 'action'),
                InputVar(tf.float32, (None,), 'reward'),
                InputVar(tf.float32, (None,) + IMAGE_SHAPE3, 'next_state'),
                InputVar(tf.bool, (None,), 'isOver') ]

    def _get_DQN_prediction(self, image):
        """ image: [0,255]"""
        image = image / 255.0
        with argscope(Conv2D, nl=PReLU.f, use_bias=True):
            return (LinearWrap(image)
                .Conv2D('conv0', out_channel=32, kernel_shape=5)
                .MaxPooling('pool0', 2)
                .Conv2D('conv1', out_channel=32, kernel_shape=5)
                .MaxPooling('pool1', 2)
                .Conv2D('conv2', out_channel=64, kernel_shape=4)
                .MaxPooling('pool2', 2)
                .Conv2D('conv3', out_channel=64, kernel_shape=3)

                # the original arch
                #.Conv2D('conv0', image, out_channel=32, kernel_shape=8, stride=4)
                #.Conv2D('conv1', out_channel=64, kernel_shape=4, stride=2)
                #.Conv2D('conv2', out_channel=64, kernel_shape=3)

                .FullyConnected('fc0', 512, nl=lambda x, name: LeakyReLU.f(x, 0.01, name))
                .FullyConnected('fct', NUM_ACTIONS, nl=tf.identity)())

    def _build_graph(self, inputs):
        state, action, reward, next_state, isOver = inputs
        self.predict_value = self._get_DQN_prediction(state)
        action_onehot = tf.one_hot(action, NUM_ACTIONS, 1.0, 0.0)
        pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)    #N,
        max_pred_reward = tf.reduce_mean(tf.reduce_max(
            self.predict_value, 1), name='predict_reward')
        add_moving_summary(max_pred_reward)

        with tf.variable_scope('target'):
            targetQ_predict_value = self._get_DQN_prediction(next_state)    # NxA

        # DQN
        #best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,

        # Double-DQN
        tf.get_variable_scope().reuse_variables()
        next_predict_value = self._get_DQN_prediction(next_state)
        self.greedy_choice = tf.argmax(next_predict_value, 1)   # N,
        predict_onehot = tf.one_hot(self.greedy_choice, NUM_ACTIONS, 1.0, 0.0)
        best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)

        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * GAMMA * tf.stop_gradient(best_v)

        cost = symbf.huber_loss(target - pred_action_value)
        summary.add_param_summary([('conv.*/W', ['histogram', 'rms']),
                                   ('fc.*/W', ['histogram', 'rms']) ])   # monitor all W
        self.cost = tf.reduce_mean(cost, name='cost')

    def update_target_param(self):
        vars = tf.trainable_variables()
        ops = []
        for v in vars:
            target_name = v.op.name
            if target_name.startswith('target'):
                new_name = target_name.replace('target/', '')
                logger.info("{} <- {}".format(target_name, new_name))
                ops.append(v.assign(tf.get_default_graph().get_tensor_by_name(new_name + ':0')))
        return tf.group(*ops, name='update_target_network')

    def get_gradient_processor(self):
        return [MapGradient(lambda grad: \
                tf.clip_by_global_norm([grad], 5)[0][0]),
                SummaryGradient()]

def get_config():
    logger.auto_set_dir()

    M = Model()
    dataset_train = ExpReplay(
            predictor_io_names=(['state'], ['fct/output']),
            player=get_player(train=True),
            batch_size=BATCH_SIZE,
            memory_size=MEMORY_SIZE,
            init_memory_size=INIT_MEMORY_SIZE,
            exploration=INIT_EXPLORATION,
            end_exploration=END_EXPLORATION,
            exploration_epoch_anneal=EXPLORATION_EPOCH_ANNEAL,
            update_frequency=4,
            reward_clip=(-1, 1),
            history_len=FRAME_HISTORY)

    lr = tf.Variable(0.001, trainable=False, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
        callbacks=Callbacks([
            StatPrinter(),
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate',
                [(150, 4e-4), (250, 1e-4), (350, 5e-5)]),
            HumanHyperParamSetter('learning_rate', 'hyper.txt'),
            HumanHyperParamSetter(ObjAttrParam(dataset_train, 'exploration'), 'hyper.txt'),
            RunOp(lambda: M.update_target_param()),
            dataset_train,
            PeriodicCallback(Evaluator(EVAL_EPISODE, ['state'], ['fct/output']), 3),
        ]),
        # save memory for multiprocess evaluator
        session_config=get_default_sess_config(0.6),
        model=M,
        step_per_epoch=STEP_PER_EPOCH,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    parser.add_argument('--task', help='task to perform',
            choices=['play', 'eval', 'train'], default='train')
    parser.add_argument('--rom', help='atari rom', required=True)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.task != 'train':
        assert args.load is not None
    ROM_FILE = args.rom

    if args.task != 'train':
        cfg = PredictConfig(
                model=Model(),
                session_init=SaverRestore(args.load),
                input_var_names=['state'],
                output_var_names=['fct/output:0'])
        if args.task == 'play':
            play_model(cfg)
        elif args.task == 'eval':
            eval_model_multithread(cfg, EVAL_EPISODE)
    else:
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        QueueInputTrainer(config).train()

