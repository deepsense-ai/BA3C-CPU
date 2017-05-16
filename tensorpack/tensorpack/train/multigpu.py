#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: multigpu.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import numpy as np
import copy
import random
from collections import defaultdict

import tensorflow as tf
import itertools, re
from six.moves import zip, range
from tensorflow.python.client import timeline

from ..models import TowerContext
from ..utils import *
from ..utils.concurrency import LoopThread
from ..tfutils.summary import summary_moving_average
from ..tfutils.modelutils import describe_model
from ..tfutils import *

from .trainer import QueueInputTrainer

__all__ = ['AsyncMultiGPUTrainer', 'SyncMultiGPUTrainer']

class MultiGPUTrainer(QueueInputTrainer):
    """ Base class for multi-gpu training"""
    def __init__(self, config, input_queue=None, predict_tower=None):
        super(MultiGPUTrainer, self).__init__(config, input_queue, predict_tower)
        assert len(config.tower) >= 1, "MultiGPUTrainer must be used with at least one GPU."
        self.dequed_inputs = []
        self.dummy = config.extra_arg['dummy']
        print 'MultiGPUTrainer __init__ dummy = {dummy}'.format(dummy=self.dummy)

    @staticmethod
    def _average_grads(tower_grads):
        ret = []
        with tf.name_scope('AvgGrad'):
            for grad_and_vars in zip(*tower_grads):
                v = grad_and_vars[0][1]
                try:
                    grad = tf.add_n([x[0] for x in grad_and_vars]) / float(len(tower_grads))
                except:
                    logger.error("Error while processing gradients of {}".format(v.name))
                    raise
                ret.append((grad, v))
        return ret

    def _multi_tower_grads(self):
        logger.info("Training a model of {} tower".format(
            len(self.config.tower)))

        grad_list = []
        for idx, t in enumerate(self.config.tower):
            if self.config.extra_arg['cpu'] == 1:
                dev = '/cpu:{}'.format(t)
            else:
                dev = '/gpu:{}'.format(t)
            with tf.device(dev), \
                    TowerContext('tower{}'.format(idx)) as scope:
                logger.info("Building graph for training tower {idx}..., {dev}".format(idx=idx, dev=dev))

                # IMPORTANT(maciek): Real or Fake?
                if self.dummy:
                    import numpy as np
                    el1 = tf.ones((128, 84, 84, 12), dtype=tf.float32, name='el1')
                    el2 = tf.constant(np.random.randint(0, 3, size=(128,)), dtype=tf.int32, name='el2')
                    el3 = tf.random_normal((128,), dtype=tf.float32, name='el2')
                    model_inputs = (el1, el2, el3)
                else:
                    model_inputs = self._get_model_inputs()    # each tower dequeue from input queue

                self.dequed_inputs.append(model_inputs)
                #import pdb; pdb.set_trace()
                self.model.build_graph(model_inputs)
                cost_var = self.model.get_cost() # build tower

                # TODO gate_gradienst=0 seems to be faster?
                grad_list.append(
                    self.config.optimizer.compute_gradients(cost_var, gate_gradients=0))

                if idx == 0:
                    tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, cost_var)
                    tf.get_variable_scope().reuse_variables()
                    # avoid repeated summary from each device
                    backup = backup_collection(SUMMARY_BACKUP_KEYS)

        restore_collection(backup)
        return grad_list

class SyncMultiGPUTrainer(MultiGPUTrainer):
    def train(self):
        self.elapsed_times = defaultdict(list)
        self.init_session_and_coord()
        self._build_enque_thread()

        grad_list = self._multi_tower_grads()

        grads = MultiGPUTrainer._average_grads(grad_list)
        grads = self.process_grads(grads)

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            summary_moving_average(), name='train_op')
        describe_model()

        # [debug]: do nothing in training
        #self.train_op = self.dequed_inputs[0][0] + self.dequed_inputs[1][0]
        self.main_loop()


class AsyncMultiGPUTrainer(MultiGPUTrainer):
    def train(self):
        self.init_session_and_coord()
        self._build_enque_thread()

        grad_list = self._multi_tower_grads()
        # pretend to average the grads, in order to make async and
        # sync have consistent effective learning rate
        def scale(grads):
            with tf.name_scope('AsyncScaleGrad'):
                return [(grad / len(self.config.tower) if grad is not None else None, var)
                            for grad, var in grads]
        grad_list = map(scale, grad_list)
        grad_list = [self.process_grads(g) for g in grad_list]

        # use grad from the first tower for iteration in main thread
        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grad_list[0], get_global_step_var()),
            summary_moving_average(), name='train_op')
        describe_model()

        self._start_async_threads(grad_list)
        self.step_counter = 0
        self.main_thread_counter = 0
        self.step_times = []

        self.main_loop()


    def _start_async_threads(self, grad_list):
        import threading
        # prepare train_op for the rest of the towers
        # itertools.count is atomic w.r.t. python threads
        self.async_step_counter = itertools.count()
        self.training_threads = []


        queue_size_op = self.input_queue.size()
        self.elapsed_times = defaultdict(list)

        for k in range(1, len(self.config.tower)):
            train_op = self.config.optimizer.apply_gradients(grad_list[k])

            def f(op=train_op, idx=k): # avoid late-binding
                oper_id = random.randint(0, 10000)
                # print 100 * 'KURWA'
                # res = self.sess.run(self.dequed_inputs[0])
                # for el in res:
                #     print el.shape, el.dtype

                queue_size = self.sess.run([queue_size_op])
                print 'Queueu_size', queue_size
                if idx <= self.config.extra_arg['threads_to_trace']:
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    timer = 0
                    self.sess.run([op], options=run_options, run_metadata=run_metadata)
                    elapsed_time = 0
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline_intra_{intra_op_par}_nrthreads_{nr_threads}_thidx_{th_idx}_{oper_id}.json'.format(
                            intra_op_par=self.config.extra_arg['intra_op_par'],
                            #inter_op_par=self.config.extra_arg['inter_op_par'],
                            nr_threads=len(self.config.tower),
                            th_idx=idx,
                            oper_id=len(self.elapsed_times[idx])), 'w') as f:
                        f.write(ctf)
                else:
                    timer = 0
                    self.sess.run([op])
                    elapsed_time = 0


                print 'Completed train in thread, id={idx}, {oper_id}, {elapsed_time}'.format(
                    idx=idx, elapsed_time=elapsed_time,
                    oper_id=len(self.elapsed_times[idx])
                )
                self.elapsed_times[idx].append(elapsed_time)
                #print 'completed', idx, oper_id
                next(self.async_step_counter)

            th = LoopThread(f)
            th.pause()
            th.start()
            self.training_threads.append(th)
        self.async_running = False

    def run_step(self):
        print 'Runstep!, ', self.step_counter
        print 'self.async_running', self.async_running

        if not self.async_running:
            self.async_running = True
            for th in self.training_threads: # resume all threads
                th.resume()
        next(self.async_step_counter)

        if self.config.extra_arg['max_steps'] is not None and self.step_counter >= self.config.extra_arg['max_steps']:
            import os
            import signal
            os.killpg(os.getpgrp(), signal.SIGKILL)

            import sys
            sys.exit()
        else:
            self.step_counter += 1

        s = ("Q-debug id=dkjs, tf_queue_size {qsize}".format(qsize=self.queue_size_op.eval()))
        # KURWA: LOGMAC
        logger.debug(s)

        super(AsyncMultiGPUTrainer, self).run_step()

        self.main_thread_counter += 1
        elapsed_time = 0
        self.step_times.append(elapsed_time)
        last_k = 20
        mean_step_time = np.mean(self.step_times[-last_k:])

    def _trigger_epoch(self):
        self.async_running = False
        for th in self.training_threads:
            th.pause()
        try:
            async_step_total_cnt = int(re.findall(
                '[0-9]+', self.async_step_counter.__str__())[0])
            self.write_scalar_summary(
                    'async_global_step', async_step_total_cnt)
        except:
            pass
        super(AsyncMultiGPUTrainer, self)._trigger_epoch()
