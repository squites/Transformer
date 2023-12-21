import torch
import torch.nn as nn
from torch.nn import functional as F


""" Operations to apply on tensors in order to perform parallelism"""

# splits several elements and scatters them on each device
class Scatter(nn.Module):
    def __init__(self, devices):
        self.devices = devices

    def forward(self, tensor):
        tensor = nn.parallel.scatter(self.devices, tensor)
        return tensor

# gather tensors existing in multiple devices into one
class Gather(nn.Module):
    def __init__(self):
    # TODO

# performs a specific operation with the data held by each device and gathers the output into one device
class Reduce(nn.Module):
    def __init__(self):
    # TODO

# copies data from one device to all devices
class Broadcast(nn.Module): 
    def __init__(self):
        # TODO

# 
class AllReduce(nn.Module):
    def __init__(self):
        # TODO

#
class AllGather(nn.Module):
    def __init__(self):
        # TODO

#
class ReduceScatter(nn.Module):
    def __init__(self):
        # TODO

#
class Barrier(nn.Module):
    def __init__(self):
        # TODO

# data parallelism
class DataParallelism(nn.Module):
    """ Data parallelism module where performs forward and backward with data in parallel across multiple devices"""
    def forward():
    # 1st way: problem: gpu memory imbalance because all the logits are concentrated in one gpu
    # scatter
    # broadcast
    # logits
    # gather logits: instead of gathering logits, it's better to compute the loss on each device then gather the loss
    # compute loss

    # Alternative way to solve the problem above: since loss is scalar rather than tensors in logits, so it can alleviate a bit
    # scatter
    # broadcast
    # logits
    # instead of gathering logits, it's better to compute the loss on each device then gather the loss 
    # reduction on the gathered losses (sum or mean)

    def backward():
    # scatter loss
    # calculate gradients
    # reduce
    # update model