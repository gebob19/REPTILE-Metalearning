import torch.nn as nn 
import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from functools import partial

import numpy as np
import tensorflow as tf
DEFAULT_OPTIMIZER = partial(tf.train.AdamOptimizer, beta1=0)

class OmniglotModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self. n_classes = n_classes

        conv_block = lambda in_dim:(nn.Conv2d(in_dim, 64, 3, stride=2, padding=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU())

        self.conv1 = nn.Conv2d(1, 64, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.rl1 = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.rl2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.rl3 = nn.ReLU()

        self.conv4 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.rl4 = nn.ReLU()
        
        # self.cnn = nn.Sequential(
        #     *conv_block(1),
        #     *conv_block(64),
        #     *conv_block(64), 
        #     *conv_block(64)
        # )
        self.linear = nn.Sequential(
            nn.Linear(256, n_classes)
        )
        
    def forward(self, x):
        self.convs = []

        x = self.conv1(x)
        # tmp1 = x 
        x = self.bn1(x)
        # tmp2 = x
        x = self.rl1(x)
        # tmp3 = x
        self.convs.append(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.rl2(x)

        self.convs.append(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.rl3(x)

        self.convs.append(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.rl4(x)

        self.convs.append(x)

        # x = self.cnn(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear(x)
        return x, self.convs#tmp1, tmp2, tmp3

    def clone(self):
        clone = OmniglotModel(self.n_classes)
        clone.load_state_dict(self.state_dict())
        return clone.to(device)


# 
class TFOmniglotModel:
    """
    A model for Omniglot classification.
    """
    def __init__(self, num_classes, lr, optimizer=DEFAULT_OPTIMIZER):
        self.input_ph = tf.placeholder(tf.float32, shape=(None, 28, 28))
        out = tf.reshape(self.input_ph, (-1, 28, 28, 1))

        for _ in range(4):
            out = tf.layers.conv2d(out, 64, 3, strides=2, padding='same')
            out = tf.layers.batch_normalization(out, training=True)
            out = tf.nn.relu(out)
        # flatten
        out = tf.reshape(out, (-1, int(np.prod(out.get_shape()[1:]))))

        self.logits = tf.layers.dense(out, num_classes)
        self.label_ph = tf.placeholder(tf.int32, shape=(None,))
        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_ph,
                                                                   logits=self.logits)
        self.predictions = tf.argmax(self.logits, axis=-1)
        self.minimize_op = optimizer(lr).minimize(self.loss)