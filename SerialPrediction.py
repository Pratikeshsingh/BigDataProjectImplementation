#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  29 10:54:59 2020

@author: pratikeshsingh
"""

from multiprocessing import Pool, Process
import psutil
import sys
import tensorflow as tf
import time

num_cpus = 1
filename = '/tmp/model'
num_trials=3

class Model(object):
    def __init__(self):
        # Load the model and some data.
        self.model = tf.keras.models.load_model(filename)
        mnist = tf.keras.datasets.mnist.load_data()
        self.x_test = mnist[1][0] / 255.0

    def evaluate_next_batch(self):
        return self.model.predict(self.x_test)

print('Using {} cores.'.format(num_cpus))
actor = Model()

durations = []
for _ in range(num_trials):
    start_time = time.time()

    for j in range(10):
        results = [actor.evaluate_next_batch() ]

    duration = time.time() - start_time
    durations.append(duration)
    print('Serial Prediction took {} seconds.'.format(duration))

print('Uses {} cores.'.format(num_cpus))