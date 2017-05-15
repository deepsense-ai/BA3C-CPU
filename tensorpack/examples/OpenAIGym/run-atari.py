#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# File: run-atari.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import tensorflow as tf
import os, sys, re, time
import random
import argparse
import six

from tensorpack import *
from tensorpack.RL import *

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
GAMMA = 0.99
CHANNEL = FRAME_HISTORY * 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

NUM_ACTIONS = None
ENV_NAME = None

from common import play_one_episode

def get_player(dumpdir=None):
    pl = GymEnv(ENV_NAME, dumpdir=dumpdir, auto_restart=False)
    pl = MapPlayerState(pl, lambda img: cv2.resize(img, IMAGE_SIZE[::-1]))

    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_action_space().num_actions()

    pl = HistoryFramePlayer(pl, FRAME_HISTORY)
    return pl

class Model(ModelDesc):
    def _get_input_vars(self):
        assert NUM_ACTIONS is not None
        return [InputVar(tf.float32, (None,) + IMAGE_SHAPE3, 'state'),
                InputVar(tf.int32, (None,), 'action'),
                InputVar(tf.float32, (None,), 'futurereward') ]

    def _get_NN_prediction(self, image):
        image = image / 255.0
        
        print(CHANNEL)
        dummy_channels = 16 - CHANNEL
        image = tf.concat(3, [image, tf.zeros((tf.shape(image)[0], 84, 84, dummy_channels))])
        with argscope(Conv2D, nl=tf.nn.relu):
            input_sum = tf.reduce_sum(tf.abs(image))
            l = Conv2D('conv0', image, out_channel=32, kernel_shape=5, use_bias=False, padding='VALID')
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, out_channel=32, kernel_shape=5, use_bias=False, padding='VALID')
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, out_channel=64, kernel_shape=5, use_bias=False, padding='VALID')

            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, out_channel=64, kernel_shape=3, use_bias=False, padding='VALID')

        l = FullyConnected('fc0', l, 512, nl=tf.identity)
        
        l = PReLU('prelu', l)
        policy = FullyConnected('fc-pi', l, out_dim=NUM_ACTIONS, nl=tf.identity)
        return policy

    
#    def _get_NN_prediction(self, image):
#        image = image / 255.0
#        with argscope(Conv2D, nl=tf.nn.relu):
#            l = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
#            l = MaxPooling('pool0', l, 2)
#            l = Conv2D('conv1', l, out_channel=32, kernel_shape=5)
#            l = MaxPooling('pool1', l, 2)
#            l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
#            l = MaxPooling('pool2', l, 2)
#            l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)
#
#        l = FullyConnected('fc0', l, 512, nl=tf.identity)
#        l = PReLU('prelu', l)
#        policy = FullyConnected('fc-pi', l, out_dim=NUM_ACTIONS, nl=tf.identity)
#        return policy

    def _build_graph(self, inputs):
        state, action, futurereward = inputs
        policy = self._get_NN_prediction(state)
        self.logits = tf.nn.softmax(policy, name='logits')

def run_submission(cfg):
    dirname = 'gym-submit'
    player = get_player(dumpdir=dirname)
    predfunc = get_predict_func(cfg)
    for k in range(100):
        if k != 0:
            player.restart_episode()
        score = play_one_episode(player, predfunc)
        print("Total:", score)

def do_submit():
    dirname = 'gym-submit'
    gym.upload(dirname, api_key='xxx')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model', required=True)
    parser.add_argument('--env', help='env', required=True)
    args = parser.parse_args()

    ENV_NAME = args.env
    p = get_player(); del p    # set NUM_ACTIONS

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cfg = PredictConfig(
            model=Model(),
            session_init=SaverRestore(args.load),
            input_var_names=['state'],
            output_var_names=['logits'])
    run_submission(cfg)
