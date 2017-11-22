#! /usr/bin/env python

import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import data_loader

# Parameters
# ==================================================
#ftype = torch.FloatTensor
#ltype = torch.LongTensor
ftype = torch.cuda.FloatTensor
ltype = torch.cuda.LongTensor

# Data loading params
glove_file = "../dataset/glove.6B.300d.txt"
train_file = "../dataset/tasksv12/en-10k/qa1_single-supporting-fact_train.txt"
test_file = "../dataset/tasksv12/en-10k/qa1_single-supporting-fact_test.txt"

# Model Hyperparameters
word_dim = 300
word_maxlen = 68 # 20400 / 300
sent_maxlen = 10
ques_maxlen = 4
answer_cnt = 1
TM = 3
alpha = 1
beta = 0

# Training Parameters
batch_size = 10
num_epochs = 30
learning_rate = 0.0001
momentum = 0.9
evaluate_every = 1

# Data Preparation
# ===========================================================
# Load data
print("Loading data...")
train_dict, train_input, train_input_len, train_question, train_question_len, train_target, train_gate, test_input, test_input_len, test_question, test_question_len, test_target, test_gate= data_loader.load_data(glove_file, train_file, test_file)

print("Train/Test/Word: {:d}/{:d}/{:d}".format(len(train_input), len(test_input), len(train_dict)))
print("==================================================================================")

class InputModule(nn.Module):
    def __init__(self, input_size, hidden_size, maxlen):
        super(InputModule, self).__init__()

        # attributes:
        self.maxlen = maxlen
        self.num_layers = 1
        self.input_size = input_size
        self.hidden_size = hidden_size

        # modules:
        self.gru = nn.GRU(self.input_size, self.hidden_size, num_layers=self.num_layers)

    def forward(self, input_seq, input_len):
        output, hx = self.gru(input_seq)
        output = output.permute(1,0,2)

        # (batch) x (maxlen) x (embed_dim)
        ret = []
        for i, batch in enumerate(output):
            ret.append(batch[input_len[i]].view(1, self.maxlen, word_dim))
        return torch.cat(ret, 0)

class EpisodicModule(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EpisodicModule, self).__init__()

        # attributes:
        self.Wb = Variable(torch.randn(word_dim, batch_size)).type(ftype)
        self.att_input = word_dim * 9
        self.att_hidden = 16

        self.num_layers = 1
        self.gru_input = input_size
        self.gru_hidden = hidden_size

        # modules:
        self.linear1 = nn.Linear(self.att_input, self.att_hidden)
        self.linear2 = nn.Linear(self.att_hidden, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.gru = nn.GRU(self.gru_input, self.gru_hidden, num_layers=self.num_layers)

    def forward(self, cs, m, q, mask):
        G = []
        for c in cs:
            # (embed_dim * 9) x (embed_dim)
            z = torch.cat([c,m,q, torch.mul(c,q), torch.mul(c,m), torch.abs(c-q), torch.abs(c-m), 
                torch.mm(torch.mm(c, self.Wb), q), torch.mm(torch.mm(c, self.Wb), m)]
                , 1)
            G.append(self.sigmoid(self.linear2(self.tanh(self.linear1(z)))).view(1, -1))
        # (sent_maxlen) x (batch)
        G = torch.t(torch.mul(torch.cat(G, 0), mask))
        G_soft = torch.t(self.softmax(G))
        
        output = []
        for i in xrange(len(cs)):
            output.append(torch.mul(G_soft[i], torch.t(cs[i])).view(1, -1, word_dim))
        # (sent_maxlen) x (batch) x (embed_dim)
        output = torch.cat(output, 0)

        _, m = self.gru(output, m.view(1, -1, word_dim))

        return G, m.view(-1, word_dim)

class AnswerModule(nn.Module):
    def __init__(self, input_size, word_cnt, question_dim):
        super(AnswerModule, self).__init__()

        # attributes:
        self.gru_hidden = input_size
        self.word_cnt = word_cnt
        self.question_dim = question_dim
        self.gru_input = word_cnt + question_dim

        # moduels:
        self.linear = nn.Linear(input_size, word_cnt, bias=False)
        self.softmax = nn.Softmax()
        self.grucell = nn.GRUCell(self.gru_input, self.gru_hidden)

    def forward(self, m_T, q):
        a = m_T 
        output = []
        for _ in xrange(answer_cnt):
            W_a = self.linear(a)
            y_t = self.softmax(W_a)
            a = self.grucell(torch.cat([y_t, q], 1), a)
            output.append(W_a)

        # How to handel muliple answer
        return output[0]

def parameters():

    params = []
    for model in [input_model, question_model, episodic_model, answer_model]:
        params += list(model.parameters())

    return params

def make_mask(maxlen, dim, length):
    one = [1]*dim
    zero = [0]*dim
    mask = []
    for c in length:
        for r in c:
            mask.append(one if r > 0 else zero)
            #mask.append(one*len(c) + zero*(maxlen-len(c)))

    # (batch) * maxlen * dim 
    # [[1 1 1 ... 1 0 0 0 ... 0]...]
    return Variable(torch.from_numpy(np.asarray(mask)).type(ftype), requires_grad=False)

def run(input_seq, input_len, question, question_len, target, gate, step):

    optimizer.zero_grad()

    # (batch) x (word_maxlen) x (embed_dim) -> (word_maxlen) * (batch) * (embed_dim)
    input_seq = Variable(torch.from_numpy(np.asarray(input_seq))).type(ftype)
    input_seq = input_seq.view(-1, word_maxlen, word_dim).permute(1,0,2)

    # (batch) x (sent_maxlen)
    input_len = torch.from_numpy(np.asarray(input_len)).type(ltype)
    for length in input_len:
        length[0] -= 1
        for i in xrange(1, len(length)):
            if length[i] != 0:
                length[i] += length[i-1] 
    # (batch) x 1
    question_len = torch.from_numpy(np.asarray(question_len)).type(ltype).view(-1, 1) -1

    # (batch) x (qeus_maxlen) x (embed_dim) -> (ques_maxlen) x (batch) x (embed_dim)
    question = Variable(torch.from_numpy(np.asarray(question))).type(ftype)
    question = question.view(-1, ques_maxlen, word_dim).permute(1,0,2)

    # (batch) * (targ_maxlen) x (embed_dix)
    target = Variable(torch.from_numpy(np.asarray(target))).type(ltype).view(-1)
    gate = Variable(torch.from_numpy(np.asarray(gate))).type(ltype).view(-1)

    # (word_maxlen) * (batch) * (embed_dim) -> (sent_maxlen) x (batch) x (embed_dim)
    input_output = input_model(input_seq, input_len)
    io_mask = make_mask(sent_maxlen, word_dim, input_len).view(-1, sent_maxlen, word_dim)
    input_output = torch.mul(input_output, io_mask).permute(1,0,2)
    # (ques_maxlen) * (batch) * (embed_dim) -> (batch) x (embed_dim)
    quest_output = torch.squeeze(question_model(question, question_len))
    # (sent_maxlen) x (batch) x (embed_dim) -> (sent_maxlen) x (batch) x (embed_dim)
    episodic_output = quest_output
    io_mask = torch.t(make_mask(sent_maxlen, 1, input_len).view(-1, sent_maxlen))
    for _ in xrange(TM):
        g, episodic_output = episodic_model(input_output, episodic_output, quest_output, io_mask)
    # (sent_maxlen) x (batch) x (embed_dim) -> (batch) x (word_cnt)
    answer_output = answer_model(episodic_output, quest_output)

    Eans = loss_model(answer_output, target)
    Egat = loss_model(g, gate)

    J = alpha * Egat + beta * Eans

    answer_output = np.argmax(answer_output.data.cpu().numpy(), axis=1)
    target = target.data.cpu().numpy()
    hit_cnt = np.sum(np.array(answer_output) == np.array(target))

    if step > 1:
        return hit_cnt, J.data.cpu().numpy()

    J.backward()
    optimizer.step()
    
    return hit_cnt, J.data.cpu().numpy()

def print_score(batches, step):
    total_hit_cnt = 0.
    total_loss = 0.0

    for batch in batches:
        batch_input, batch_input_len, batch_question, batch_question_len, batch_target, batch_gate = zip(*batch)
        hit_cnt, batch_loss = run(batch_input, batch_input_len, batch_question, batch_question_len, batch_target, batch_gate, step=step)
        total_loss += batch_loss
        total_hit_cnt += hit_cnt

    print("loss: ", total_loss/len(test_target))
    print("acc: ", total_hit_cnt/len(test_target)*100)

###############################################################################################
input_model = InputModule(word_dim, word_dim, sent_maxlen).cuda()
question_model = InputModule(word_dim, word_dim, 1).cuda()
episodic_model = EpisodicModule(word_dim, word_dim).cuda()
answer_model = AnswerModule(word_dim, len(train_dict), word_dim).cuda()
loss_model = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(parameters(), lr=learning_rate, momentum=momentum)

for i in xrange(num_epochs):
    if i == 5:
        beta = 1
    # Training
    train_batches = data_loader.batch_iter(list(zip(train_input, train_input_len, train_question, train_question_len, train_target, train_gate)), batch_size)
    total_loss = 0.
    total_hc = 0.
    for j, train_batch in enumerate(train_batches):
        batch_input, batch_input_len, batch_question, batch_question_len, batch_target, batch_gate = zip(*train_batch)
        batch_hc, batch_loss = run(batch_input, batch_input_len, batch_question, batch_question_len, batch_target, batch_gate, step=1)
        total_hc +=batch_hc
        total_loss +=batch_loss
        if (j+1) % 200 == 0:
            print("batch #{:d}: ".format(j+1)), "batch_loss :", total_loss/j, "acc. :", total_hc/batch_size/j*100, datetime.datetime.now()

    # Evaluation
    if (i+1) % evaluate_every == 0:
        print("==================================================================================")
        print("Evaluation at epoch #{:d}: ".format(i+1))
        test_batches = data_loader.batch_iter(list(zip(test_input, test_input_len, test_question, test_question_len, test_target, test_gate)), batch_size)
        print_score(test_batches, step=2)

input_model.eval()
question_model.eval()
episodic_model.eval()
answer_model.eval()
loss_model.eval()

# Testing
print("Training End..")
print("==================================================================================")
print("Test: ")
test_batches = data_loader.batch_iter(list(zip(test_input, test_input_len, test_question, test_question_len, test_target, test_gate)), batch_size)
print_score(test_batches, step=3)
