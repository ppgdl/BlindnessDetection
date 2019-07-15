#!/usr/bin/env python
# encoding: utf-8
import matplotlib.pyplot as plt
from PIL import Image
import os

def plot_loss(save_dir, x, y):
#    print(loss_log.shape)
    plt.figure()
#    for key, value in y.items():
#        plt.plot(x, value, label=key)
    plt.plot(x,y)
    plt.legend(loc='upper right')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(save_dir, bbox_inches='tight')
    plt.show()
