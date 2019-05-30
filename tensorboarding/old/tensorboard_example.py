from time import sleep

import torch
from tensorboardX import SummaryWriter

import numpy as np

writer = SummaryWriter('tensorboard_logs/first_tensorboarding')

for i in range(100):
    sleep(1)
    writer.add_scalar('y=x**2', i ** 2, i)
    writer.add_scalar('y=xsinx':i*np.sin(i/5), i)

writer.close()