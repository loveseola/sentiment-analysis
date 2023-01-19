
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import re
import jieba
import csv
import openpyxl
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
import torch
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
import gensim
import time
import traceback
from datasets import load_ylk,load_tt,build_word2vec,build_word2id,text_to_array,Data_set

class BILSTMModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, pretrained_weight,hidden_dim, num_layers,num_classes):

        super(BILSTMModel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding.from_pretrained(pretrained_weight)       #对于自建单词词向量：nn.Embedding(vocab_size, embedding_dim)词嵌入层：nn.Embedding(num_embeddings, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=False,bidirectional=True)
        self.fc = nn.Linear(4*hidden_dim,num_classes)#线性全连接层，输入的tensor的数量（隐藏层乘以4,并行拼了起来），输出是num_classes
        #self.dropout = nn.Dropout(0.5)#可以加或者不加

    def forward(self, x):
        #这里我的x 形状是batch_size,embedding_dim
        #nn.Embedding(num_embeddings, embedding_dim) embedding的字典大小和每个词用多少维的向量表示
        embedding = self.embedding(x)#必有：提取词特征，输出形状为[batch, seq_len, embedding_dim][32, 721, 300]
        #output 形状是(seq_len, batch_size, D*hidden_dim)[721, 32, 128]，final_hidden_state.shape 是（D*numlayer,batch_size,hidden_dim）
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding.permute([1, 0, 2]))#必有:rnn.LSTM只返回最后一层的隐藏层在各时间步的隐藏状态。
        #encoding shape: (batch_size, 2*D*hidden_dim)[32, 256]
        encoding = torch.cat([output[0], output[-1]], dim=1)#按照行并排起来,所以*2,output[0]的shape为(batch,D*hidden_dim)
        #outs:(batch_size,n_class)[32,2]
        outs = self.fc(encoding)
        return outs

class BiLSTM_Attention(nn.Module):

    def __init__(self, vocab_size, embedding_dim, pretrained_weight,hidden_dim, num_layers,num_classes):

        super(BiLSTM_Attention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding.from_pretrained(pretrained_weight)       #对于自建单词词向量：nn.Embedding(vocab_size, embedding_dim)词嵌入层：nn.Embedding(num_embeddings, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=False,bidirectional=True)
        self.fc = nn.Linear(2*hidden_dim,num_classes)
        #self.dropout = nn.Dropout(0.5)#可以加或者不加

        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        # What is nn. Parameter(data(Tensor)=None, requires_grad=True) ? Explain
        # 理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter.
        # 所以经过类型转换这个Tensor变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, hidden_dim * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim * 2, 1))
        #self.decoder2 = nn.Linear(2*hidden_dim, 2)
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, x):
        #这里我的x 形状是batch_size,embedding_dim
        #nn.Embedding(num_embeddings, embedding_dim) embedding的字典大小和每个词用多少维的向量表示
        embedding = self.embedding(x)#必有：提取词特征，输出形状为[batch, seq_len, embedding_dim]
        #output 形状是batch_size, seq_len, D * hidden_dim,final_hidden_state shape是（D * num_layers,seq_len,hidden_dim）
        output, (final_hidden_state, final_cell_state) = self.rnn(embedding.permute([0, 1, 2]))#必有:rnn.LSTM只返回最后一层的隐藏层在各时间步的隐藏状态。
        #Attention过程
        u = torch.tanh(torch.matmul(output, self.w_omega))         #[batch_size, seq_len, hidden_dim*2]
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)                     #仍为(batch_size, seq_len, 1)
        scored_x = output * att_score                              #[batch_size, seq_len, hidden_dim*2]
        #结束
        context = torch.sum(scored_x, dim=1)                  #[batch_size, hidden_dim*2]
        outs = self.fc(context)                          ## out形状是(batch_size, 2)
        return outs
'''
if __name__ == "__main__":
    embedding_dim=300
    hidden_dim=64
    num_layers=5
    batch_size=32
    num_classes=2
    split = 0.2
    stopwords_path = "./data/stopwords.txt"
    pos_data_path = "./data/neg新.xlsx"
    neg_data_path='./data/pos新.xlsx'
    model = gensim.models.KeyedVectors.load_word2vec_format('./data/sgns.renmin.word', encoding="utf-8")
    xyl_train,xyl_test,yyl_train,yyl_test,max_tokens,pos_and_neg=load_ylk(pos_data_path,neg_data_path,stopwords_path,split)
    word2id = build_word2id('./data/word2id.txt',pos_and_neg)
    train_array, train_label = text_to_array(word2id, max_tokens, xyl_train, yyl_train)  # shape:(len(train), seq_len)
    test_array, test_label = text_to_array(word2id, max_tokens, xyl_test, yyl_test)
    train_loader = Data_set(train_array, train_label)
    test_loader = Data_set(test_array, test_label)
    train_dataloader, test_dataloader = load_tt(batch_size, train_loader, test_loader)
    print('模型数据准备完毕')
    w2vec = build_word2vec(word2id, model,'./data/word2vec.txt')  # shape (70278, max_tokens)
    w2vec = torch.from_numpy(w2vec)
    w2vec = w2vec.float()
    model1 = BILSTMModel(max(word2id.values())+1, embedding_dim, w2vec, hidden_dim, num_layers,num_classes)
    model2 = BiLSTM_Attention(max(word2id.values()) + 1, embedding_dim, w2vec, hidden_dim, num_layers, num_classes)
    input_tensor = torch.tensor([i for i in range(1200)]).reshape([4, 300]) #实例化一个tensor,注意字典是1002不能超过
    out_tensor = model1.forward(input_tensor)
    print(out_tensor)  #概率分布size为torch.tensor([3,2])
    print(torch.argmax(out_tensor, dim=1))#返回行最大,3
    print(torch.argmax(out_tensor, dim=1).size()[0])
    out_tensor2 = model2.forward(input_tensor)
    print('-------------------------')
    print(out_tensor2)  #概率分布size为torch.tensor([3,2])
    print(torch.argmax(out_tensor2, dim=1))#返回行最大,3
    print(torch.argmax(out_tensor2, dim=1).size()[0])
'''