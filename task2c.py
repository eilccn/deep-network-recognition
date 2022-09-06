# Eileen Chang
# Project 5: Recognition Using Deep Networks
# Tasks 2C: build a truncated model

# IMPORT STATEMENTS
import sys
import numpy
import random
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from main import *


# READ NETWORK
# initialize new set of network and optimizers
continued_network4 = Net()
continued_optimizer4 = optim.SGD(network.parameters(), lr=learning_rate,
                                momentum=momentum)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)


# CLASS DEFINITIONS
class Submodel(mnist.Net):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override the forward method
    def forward( self, x ):
        x = F.relu( F.max_pool2d( self.conv1(x), 2 ) ) # relu on max pooled results of conv1
        x = F.relu( F.max_pool2d( self.conv2_drop( self.conv2(x)), 2 ) ) # relu on max pooled results of dropout of conv2
        return x


# initialize new set of network and optimizers
truncated_network = Submodel()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
truncated_model = Submodel().to(device)


# FUNCTION DEFINITIONS
# load the internal state of the network and optimizer when we last saved them
def load_network():
    network_state_dict = torch.load("/Users/eileenchang/computervision/project5/results/model.pth")
    truncated_network.load_state_dict(network_state_dict)

    optimizer_state_dict = torch.load("/Users/eileenchang/computervision/project5/results/optimizer.pth")
    continued_optimizer4.load_state_dict(optimizer_state_dict)







# MAIN FUNCTION
def main(argv):

    return

if __name__ == "__main__":
     main(sys.argv)