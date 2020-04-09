#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  30 11:06:03 2020

@author: pratikeshsingh
"""
from multiprocessing import Pool, Process
import psutil
import sys
import tensorflow as tf
import time

num_cpus = psutil.cpu_count(logical=False)
filename = '/tmp/model'
num_trials=3

num_cpus = psutil.cpu_count(logical=False)

filename = '/tmp/model'

def evaluate_next_batch(i):
    if sys.platform == 'linux':
        psutil.Process().cpu_affinity([i])
    model = tf.keras.models.load_model(filename)
    mnist = tf.keras.datasets.mnist.load_data()
    x_test = mnist[1][0] / 255.0
    return model.predict(x_test)

pool = Pool(num_cpus)
durations = []
print('Using {} cores.'.format(num_cpus))
start_time = time.time()
for _ in range(10):
    pool.map(evaluate_next_batch, range(num_cpus))

    duration = time.time() - start_time
    durations.append(duration)
    print('Parallel Prediction took {} seconds.'.format(duration))

print('Used {} cores.'.format(num_cpus))