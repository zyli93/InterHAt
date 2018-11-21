"""
Training function of InterpRecSys

@Author: Zeyu Li <zyli@cs.ucla.edu> or <zeyuli@g.ucla.edu>
"""

import time
import tensorflow as tf

from .gcn.utils import *
from .gcn.models import GCN
import numpy as np

# Set random seed
seed = 2018
np.random.seed(seed)
tf.set_random_seed(seed)

# Parameters
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'DEFAULT', 'Dataset string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs_gcn', 200, 'Number of epochs to train for GCN.')
flags.DEFINE_integer('epochs_fm', 200, 'Number of epochs to train for FM.')
# TODO: Add other parameters

sess = tf.Session()
# TODO: Add GPU support

# TODO: Create GCN and FM models.
# TODO: Take out initializers from models and initialize
sess.run()

for epoch in range(FLAGS.epochs):
    t = time.time()
    feed_dict = construct_feed_dict()  # TODO

