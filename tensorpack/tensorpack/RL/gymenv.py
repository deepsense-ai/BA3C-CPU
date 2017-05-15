#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: gymenv.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import time
from ..utils import logger
try:
    import gym
except ImportError:
    logger.warn("Cannot import gym. GymEnv won't be available.")

import threading

from ..utils.fs import *
from ..utils.stat import *
from .envbase import RLEnvironment, DiscreteActionSpace

__all__ = ['GymEnv']

_ALE_LOCK = threading.Lock()

class GymEnv(RLEnvironment):
    """
    An OpenAI/gym wrapper. Will auto restart.
    """
    def __init__(self, name, dumpdir=None, viz=False, auto_restart=True):
        with _ALE_LOCK:
            self.gymenv = gym.make(name)
        if dumpdir:
            mkdir_p(dumpdir)
            #self.gymenv = gym.wrappers.Monitor(env, directory)
            self.gymenv.monitor.start(dumpdir)

        self.reset_stat()
        self.rwd_counter = StatCounter()
        self.restart_episode()
        self.auto_restart = auto_restart
        self.viz = viz

    def restart_episode(self):
        self.rwd_counter.reset()
        self._ob = self.gymenv.reset()

    def finish_episode(self):
        self.stats['score'].append(self.rwd_counter.sum)

    def current_state(self):
        if self.viz:
            self.gymenv.render()
            time.sleep(self.viz)
        return self._ob

    def action(self, act):
        self._ob, r, isOver, info = self.gymenv.step(act)
        self.rwd_counter.feed(r)
        if isOver:
            self.finish_episode()
            if self.auto_restart:
                self.restart_episode()
        return r, isOver

    def get_action_space(self):
        spc = self.gymenv.action_space
        assert isinstance(spc, gym.spaces.discrete.Discrete)
        return DiscreteActionSpace(spc.n)

if __name__ == '__main__':
    env = GymEnv('Breakout-v0', viz=0.1)
    num = env.get_action_space().num_actions()

    from ..utils import *
    rng = get_rng(num)
    while True:
        act = rng.choice(range(num))
        #print act
        r, o = env.action(act)
        env.current_state()
        if r != 0 or o:
            print r, o
