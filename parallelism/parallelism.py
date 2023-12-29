import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F



""" Operations to apply on tensors in order to perform parallelism
	
	* At first this library will be able to perform the simplest data paralleism on tensors, but not mixing with transformers.
	Then this library will apply the parallelism to transformers. Maybe I'll create another repo for this.
"""

# splits several elements and scatters them on each device
class Scatter(nn.Module):
    #def __init__(self, devices):
    #    self.devices = devices

    def forward(self, x):
        x = torch.split(x, dim=0, split_size_or_sections=1) # check where to actually spit the tensor
        x = dist.scatter(out, scatter_list=list(x)) # not sure what is this out
        return x

# gather tensors existing in multiple devices into one
class Gather(nn.Module):
    def __init__(self, dim, index):
        self.dim = dim
        self.index = index
    
    def forward(self, x):
        x = torch.gather(x, self.dim, self.index)
        return x 
    # TODO

# performs a specific operation with the data held by each device and gathers the output into one device

#dist.init_process_group("nccl")
#rank = dist.get_rank()
#torch.cuda.set_device(rank)
#tensor = torch.ones(2, 2).to(torch.cuda.current_device()) * rank
# rank==0 => [[0, 0], [0, 0]]
# rank==1 => [[1, 1], [1, 1]]
# rank==2 => [[2, 2], [2, 2]]
# rank==3 => [[3, 3], [3, 3]]
# dist.reduce(tensor, op=torch.distributed.ReduceOp.SUM, dst=0)

class Reduce(nn.Module):
    def __init__(self, op):
        self.op = op # which operation to perform

    def forward(self, x):
        x = dist.reduce(x, self.op)
        return x
        

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


# instead of classes, should I implement only functions?

def scatter(tensor: torch.Tensor, dim:int) -> torch.Tensor:
    """ Scatter tensors """
    tensor = torch.distributed.Scatter(tensor, dim)
