# torch imports
import torch.nn.functional as F
import torch.nn as nn


## TODO: Complete this classifier
class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    ## TODO: Define the init function, the input params are required (for loading code in train.py to work)
    def __init__(self, input_features, hidden_dim, output_dim):
        """
        Initialize model with setting up the linear layers.
        then make input parameters so define the layers of our model.
        :param input_features: make our number for input features in our training/test data
        :param hidden_dim: define number of the nodes in hidden layer(s)
        :param output_dim:  number of the outputs we want to produce
        """
        super(BinaryClassifier, self).__init__()

        # define any initial layers, here
        self.fc1 = nn.Linear(input_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, int(hidden_dim//2))
        self.fc3 = nn.Linear(int(hidden_dim//2), output_dim)
        self.drop = nn.Dropout(0.25)
        self.sig = nn.Sigmoid()

    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        """
        displaythe forward pass for our model of the input features, x.
        :param x: the batch of the input features of the size (batch_size, input_features)
        :return: thesingle, sigmoid-activated value as the output
        """
        
        # define the feedforward behavior
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = self.sig(self.fc3(x))
        return x
    