import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        # A feed forward neural network F with ReLU activations
        # We use two hidden layers, the input_size is the 
        # word2vec dimension.
        self.attend1 = nn.Linear(input_size, hidden_size)
        self.attend_dropout1 = nn.Dropout(self.dropout_p)
        self.attend2 = nn.Linear(hidden_size, hidden_size)
        self.attend_dropout2 = nn.Dropout(self.dropout_p)

        # Compare
        # A feed forward neural network G with ReLU activations
        # the input of 1st layer is a concatenation of ai and betai.
        self.comp1 = nn.Linear(self.word_dim*2, hidden_size)
        self.comp_dropout1 = nn.Dropout(self.dropout_p)
        self.comp2 = nn.Linear(hidden_size, hidden_size)
        self.comp_dropout2 = nn.Dropout(self.dropout_p)

        # Aggregate
        # A feed forward neural network H that output the label
        # The input of the 1st layer is a concatenation of v1 and v2
        self.aggrg1 = nn.Linear(hidden_size*2, hidden_size)
        self.aggrg_dropout1 = nn.Dropout(self.dropout_p)
        self.aggrg2 = nn.Linear(hidden_size, hidden_size)
        self.aggrg_dropout2 = nn.Dropout(self.dropout_p)
        self.aggrg3 = nn.Linear(hidden_size, 3) # 3 labels c, n, e

    def forward(self, p, h):
        """
        Take premise and hypothesis as input.
        Then output the prediction
        """

        # embed each word
        p = p.split()
        h = h.split()
        p_emb = torch.zeros(len(p), self.word_dim, device=device)
        h_emb = torch.zeros(len(h), self.word_dim, device=device)
        for i, w in enumerate(p):
            p_emb[i] = self.embedding(torch.tensor(self.corpus.word2index[w],
                                                device=device))
        for j, w in enumerate(h):
            h_emb[j] = self.embedding(torch.tensor(self.corpus.word2index[w],
                                                device=device))

        # calculate e_{ij}
        fp = self.attend1(p_emb)
        fp = self.attend_dropout1(fp)
        fp = F.relu(fp)
        fp = self.attend2(fp)
        fp = self.attend_dropout2(fp)
        fp = F.relu(fp)

        fh = self.attend1(h_emb)
        fh = self.attend_dropout1(fh)
        fh = F.relu(fh)
        fh = self.attend2(fh)
        fh = self.attend_dropout2(fh)
        fh = F.relu(fh)
        # print 'fp ', fp
        # print 'fh ', fh

        # decompose matrix E
        E = torch.mm(fp, fh.view(self.hidden_size, -1))
        # print 'E ', E

        # normalized attention weight
        # beta = torch.zeros(len(p_emb), self.word_dim, device=device) 
        # alpha = torch.zeros(len(h_emb), self.word_dim, device=device)

        # eik = torch.sum(E, dim=1)
        # ekj = torch.sum(E, dim=0)
        # for i in range(len(p_emb)):
        #     for j in range(len(h_emb)):
        #         beta[i] += (torch.exp(E[i][j]) / eik[i]) * h_emb[j]

        # for j in range(len(h_emb)):
        #     for i in range(len(p_emb)):
        #         alpha[j] += (torch.exp(E[i][j]) / ekj[j]) * p_emb[i]

        # Do it vectorially
        eik = torch.sum(E, dim=1).view(-1,1)
        ekj = torch.sum(E, dim=0).view(1,-1)
        # print E.shape, eik.shape, ekj.shape, (E/eik).shape, (E/ekj).shape
        beta = torch.mm(E/eik, h_emb)
        alpha = torch.mm(torch.t(E/ekj), p_emb)
        
        # print 'alpha ', alpha
        # print 'beta ', beta
        # print 'alpha shape: ', alpha.shape
        # print 'beta shape: ', beta.shape

        # calculate comparison vectors
        v1 = self.comp1(torch.cat((p_emb, beta), dim=1))
        v1 = self.comp_dropout1(v1)
        v1 = F.relu(v1)
        v1 = self.comp2(v1)
        v1 = self.comp_dropout2(v1)
        v1 = F.relu(v1)

        v2 = self.comp1(torch.cat((h_emb, alpha), dim=1))
        v2 = self.comp_dropout1(v2)
        v2 = F.relu(v2)
        v2 = self.comp2(v2)
        v2 = self.comp_dropout2(v2)
        v2 = F.relu(v2)

        # aggregate and output the labels
        v1 = torch.sum(v1, dim=0)
        v2 = torch.sum(v2, dim=0)
        # print v1, v2
        y = self.aggrg1(torch.cat((v1,v2)))
        # print 'y ', y
        y = self.aggrg_dropout1(y)
        y = F.relu(y)
        y = self.aggrg2(y)
        y = self.aggrg_dropout2(y)
        y = F.relu(y)
        y = self.aggrg3(y)
        y = F.softmax(y, dim=0)

        return E, beta, alpha, v1, v2, y

    def initHidden(self):
        return torch.randn(1, 1, self.hidden_size, device=device)
