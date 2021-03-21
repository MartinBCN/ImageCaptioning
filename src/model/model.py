from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models
from torch import Tensor
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


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
    """
    Decoder LSTM
    """
    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int = 1,
                 drop_prob: float = 0.5):
        """
        Parameters
        ----------
        embed_size: int
            Size of the feature vector from the Encoder
        hidden_size: int
            Size of the LSTM hidden layer
        vocab_size: int
            Size of the vocabulary dictionary
        num_layers: int, default=1
            Number of LSTM layers
        drop_prob: float, default=0.5
            Dropout probability for the dropout layer following the LSTM
        """
        super(DecoderRNN, self).__init__()

        # Define the embedding layer using the Torch Embedding class
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)

        # LSTM Hyper-parameter
        self.embed_size = embed_size
        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # define the LSTM, self.lstm
        self.lstm = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True)

        # define a dropout layer, self.dropout
        self.drop_out = nn.Dropout(drop_prob)

        # Define the final, fully-connected output layer, self.fc
        self.fc = nn.Linear(self.hidden_size, vocab_size)

        # initialize the weights
        self.init_weights()

    def init_hidden(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        """
        Initialize the hidden state and cell state for a given batch size

        Parameters
        ----------
        batch_size: int

        Returns
        -------
        Tuple[Tensor, Tensor]
            Initial tensors for hidden and cell state, all values are set to zero
        """

        hidden_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)
        cell_state = torch.zeros((1, batch_size, self.hidden_size)).to(device)

        return hidden_state, cell_state

    def forward(self, features: Tensor, captions: Tensor) -> Tensor:
        """
        The forward method contains the following steps:
        1) Input Transformation:
            * Use the embedding layer to transform the captions (Tensor with long integers for words) to floats
            * Concatenate feature from CNN with captions
        2) Forward step in LSTM
        3)


        Parameters
        ----------
        features
        captions

        Returns
        -------
        Tensor
        """

        batch_size = features.shape[0]

        # --- Handle Captions ---

        """
        From project documentation: 
        
        # the first value returned by LSTM is all of the hidden states throughout
        # the sequence. the second is just the most recent hidden state
        
        # Add the extra 2nd dimension
        inputs = torch.cat(inputs).view(len(inputs), 1, -1)
        hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
        out, hidden = lstm(inputs, hidden)
        """
        # Remove the <end> word
        captions = captions[:, :-1]
        # Embed captions
        captions = self.embedding_layer(captions)

        # Concatenate feature vector and embedded captions
        x = torch.cat((features.unsqueeze(1), captions), dim=1)

        # --- LSTM step ---
        (hidden_state, cell_state) = self.init_hidden(batch_size)
        x, (hidden_state, cell_state) = self.lstm(x, (hidden_state, cell_state))
        # Shape of x: [batch_size, len(caption), hidden size]

        # pass x through a dropout layer
        x = self.drop_out(x)

        # --- Fully connected layer ---
        x = self.fc(x)
        # New shape of x: [batch_size, len(caption), vocabulary size]

        return x

    def sample(self, inputs: Tensor, states=None, max_len=20):
        """
        accepts pre-processed image tensor (inputs)
        and returns predicted sentence (list of tensor ids of length max_len)

        This is supposed to be the inference pipeline. The following checks need to run through:
        assert (type(output)==list), "Output needs to be a Python list"
        assert all([type(x)==int for x in output]), "Output should be a list of integers."
        assert all([x in data_loader.dataset.vocab.idx2word for x in output]),
            "Each entry in the output needs to correspond to an integer that indicates a token in the vocabulary."

        Parameters
        ----------
        inputs: Tensor
        states:
        max_len: int

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


def main():

    batch_size = 10
    embedding_size = 512

    # Vocabulary size from actual dataset
    vocabulary_size = 8852

    images = Tensor(batch_size, 3, 244, 244).to(device)
    captions = torch.randint(0, 200, (batch_size, 15,)).to(device)


    """
    Testing the loader with a batch size of 5 gives:
    torch.Size([5, 3, 224, 224])
    torch.Size([5, 12])
    torch.Size([5, 3, 224, 224])
    torch.Size([5, 16])
    torch.Size([5, 3, 224, 224])
    torch.Size([5, 14])
    """
    print('Shape images', images.shape)
    print('Shape captions', captions.shape)

    encoder = EncoderCNN(embedding_size).to(device)

    features = encoder(images)

    print('Shape features', features.shape)

    decoder = DecoderRNN(embedding_size, 256, vocabulary_size, 1).to(device)

    outputs = decoder(features, captions)

    print('Shape Outputs', outputs.shape)

    # The loss is calculated with these two:
    # To have a sensible loss we need to have the first dimension of the outputs matching the (only) dimension
    # of the captions. A prediction would correspond to taking the argmax over the vocabulary size dimension
    print(outputs.view(-1, vocabulary_size).shape)
    print(captions.view(-1).shape)

    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    criterion(outputs.view(-1, vocabulary_size), captions.view(-1))


if __name__ == '__main__':
    main()
