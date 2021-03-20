from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor


class EncoderCNN(nn.Module):
    """
    Pretrained CNN (Resnet 5) with new final layer as Encoder
    """
    def __init__(self, embed_size: int):
        """
        Parameters
        ----------
        embed_size: int
            Size of the embedding layer. Recommended size 512
        """
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images: Tensor):
        """
        Forward operation, batch of images -> batch of embeddings

        Parameters
        ----------
        images: Tensor
            Shape: batch_size, n_color (3), height, width

        Returns
        -------

        """
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int = 1,
                 drop_prob: float = 0.5):
        super(DecoderRNN, self).__init__()

        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # define the LSTM, self.lstm
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers)

        # define a dropout layer, self.dropout
        self.drop_out = nn.Dropout(drop_prob)

        # Define the final, fully-connected output layer, self.fc
        self.fc = nn.Linear(self.n_hidden, vocab_size)

        # initialize the weights
        self.init_weights()

    def forward(self, features: Tensor, captions: Tensor) -> (Tensor, Tuple[Tensor, Tensor]):

        hc = self.init_hidden(2)

        # Get x, and the new hidden state (h, c) from the lstm

        x, (h, c) = self.lstm(features, hc)

        # pass x through a dropout layer
        x = self.drop_out(x)

        # Stack up LSTM outputs using view
        x = x.reshape(-1, self.n_hidden)

        # put x through the fully-connected layer
        x = self.fc(x)

        # return x and the hidden state (h, c)
        return x, (h, c)

    def sample(self, inputs: Tensor, states=None, max_len=20):
        """
        accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len)

        Parameters
        ----------
        inputs
        states
        max_len

        Returns
        -------

        """
        pass

    def init_weights(self):
        """
        Initialize weights for fully connected layer

        Returns
        -------

        """

        init_range = 0.1

        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-1, 1)

    def init_hidden(self, n_seqs: int) -> Tuple[Tensor, Tensor]:
        """
        Initializes hidden state

        Parameters
        ----------
        n_seqs

        Returns
        -------

        """
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        return (weight.new(self.num_layers, n_seqs, self.hidden_size).zero_(),
                weight.new(self.num_layers, n_seqs, self.hidden_size).zero_())


if __name__ == '__main__':
    tensor = Tensor(10, 3, 244, 244)

    encoder = EncoderCNN(512)

    embed = encoder(tensor)

    print(embed.shape)
