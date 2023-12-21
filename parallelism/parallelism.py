import torch
import torch.nn as nn
from torch.nn import functional as F

# parallelism ops (scatter, broadcast, gather, reduce)
""" Operations to apply on tensors in order to perform parallelism"""
class Scatter(nn.Module):
    def __init__(self, devices):
        self.devices = devices

    def forward(self, tensor):
        tensor = nn.parallel.scatter(self.devices, tensor)
        return tensor

class Gather(nn.Module):
    def __init__(self):
    # TODO

class Reduce(nn.Module):
    def __init__(self):
    # TODO

class Broadcast(nn.Module):
    def __init__(self):
    # TODO


# data parallelism
""" Data parallelism module where performs forward and backward with data in parallel across multiple devices"""
def forward():
    # scatter
    # broadcast
    # logits
    # gather
    # compute loss

def backward():
    # scatter loss
    # calculate gradients
    # reduce
    # update model