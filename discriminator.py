import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class Discriminator(nn.Module):
    """
    A discriminator model from Parikh,etc(2016)

    A decomposible attention model for NLI
    """
    def __init__(self, input_size, hidden_size, w2v, corpus, dropout_p=0.1):
        super(Discriminator, self).__init__()

        self.word2vec = w2v
        self.corpus = corpus
        self.word_dim = input_size
        self.dropout_p =dropout_p
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(self.word2vec)

        # Attend
        # A feed forward neural network with ReLU activations
        # We use two hidden layers, the input_size is the 
        # word2vec dimension.
        self.attend1 = nn.Linear(input_size, hidden_size)
        self.attend_dropout1 = nn.Dropout(self.dropout_p)
        self.attend2 = nn.Linear(hidden_size, hidden_size)
        self.attend_dropout2 = nn.Dropout(self.dropout_p)

    def forward(self, p, h):
        """
        Take premise and hypothesis as input.
        Then output the prediction
        """

        # embed each word
        p_emb = [self.embedding(torch.tensor(self.corpus.word2index[word])) for word in p.split()]
        h_emb = [self.embedding(torch.tensor(self.corpus.word2index[word])) for word in h.split()]

        # calculate e_{ij}
        fp, fh = [], []
        for ai in p_emb:
            y1 = self.attend1(ai)
            y1 = self.attend_dropout1(y1)
            y1 = F.relu(y1)
            y2 = self.attend2(y1)
            y2 = self.attend_dropout2(y2)
            fp.append(F.relu(y2))

        for bj in h_emb:
            y1 = self.attend1(ai)
            y1 = self.attend_dropout1(y1)
            y1 = F.relu(y1)
            y2 = self.attend2(y1)
            y2 = self.attend_dropout2(y2)
            fh.append(F.relu(y2))

        # decompose matrix E
        E = torch.zeros(len(p_emb), len(h_emb))
        for i in range(len(p_emb)):
            for j in range(len(h_emb)):
                fai = p_emb[i].view(1, -1)
                fbj = h_emb[j].view(-1, 1)
                E[i][j] = torch.mm(fai, fbj)

        # normalized attention weight
        beta = torch.zeros(len(p_emb), self.word_dim) 
        alpha = torch.zeros(len(h_emb), self.word_dim)

        eik = torch.sum(E, dim=1)
        ekj = torch.sum(E, dim=0)
        for i in range(len(p_emb)):
            for j in range(len(h_emb)):
                beta[i] += (torch.exp(E[i][j]) / eik[i]) * h_emb[j]

        for j in range(len(h_emb)):
            for i in range(len(p_emb)):
                alpha[j] += (torch.exp(E[i][j]) / ekj[j]) * p_emb[i]



