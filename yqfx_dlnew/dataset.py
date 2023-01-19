import warnings
warnings.filterwarnings("ignore")
import re
import jieba 
import pandas as pd
import numpy as np
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

#语料库预处理
#语料库预处理
def processing(data_path,stopwords_path):
    '''
    :param data_path:语料库
    :param stopwords_path:停用词词典
    :return:语料库新增分词列
    '''
    stopwords = pd.read_csv(stopwords_path, header=None, quoting=csv.QUOTE_NONE, delimiter="\t")
    stopwords = stopwords[0].tolist()
    data=pd.read_excel(data_path)
    data.columns = ['content', 'label']
    zw = lambda x: ''.join(re.sub(re.compile(r'[^\u4e00-\u9fa5]'), '', str(x)))
    data['words']=data['content'].apply(zw)
    data['words']=[jieba.lcut(line) for line in data['words']]
    x1=[]
    for x in data['words']:
        x0=[]
        for word in x:
            if word not in stopwords:
                x0.append(word)
        x1.append(x0)
    for x in range(0,len(data['words'])):
        data['words'].iloc[x]=x1[x]
    return data
def load_ylk(pos_data_path,neg_data_path,stopwords_path,split):
    '''
    :param pos_data_path: 正语料库
    :param neg_data_path: 负语料库
    :param stopwords_path: 停用词词典
    :param split: 训练测试集切分比例
    :return: 训练，测试集样本
    '''
    pos_data=pd.read_excel(pos_data_path)
    neg_data = pd.read_excel(neg_data_path)
    if 'words' in list(pos_data.columns):
        pos=pos_data
    else:
        pos=processing(pos_data_path,stopwords_path)
    if 'words' in list(neg_data.columns):
        neg=neg_data
    else:
        neg=processing(neg_data_path,stopwords_path)
    pos_and_neg=np.append(pos['words'],neg['words'],axis=0)
    #构造对应标签数组
    table=np.append((np.ones(len(pos))),(np.zeros(len(neg))),axis=0)
    #切分训练测试集
    xyl_train,xyl_test,yyl_train,yyl_test = train_test_split(pos_and_neg,table,test_size=split)
    return xyl_train,xyl_test,yyl_train,yyl_test
# 数据集
class MyDataset(Dataset):
    def __init__(self, df, df1):
        self.data = []
        self.label = df1.tolist()
        for s in df.tolist():
            vectors = []
            for w in s:
                if w in model.index_to_key:
                    vectors.append(model[w])  # 将每个词替换为对应的词向量
            vectors = torch.Tensor(vectors)
            self.data.append(vectors)
    def __len__(self):
        return len(self.label)
    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

def collate_fn(data):
    """
    param data:第0维：data,第一维：label
    return 序列化的data,记录实际长度的序列，以及label列表
    """
    data.sort(key=lambda x:len(x[0]),reverse=True)# pack_padded_sequence要求要按照序列的长度倒序排列
    data_length = [len(sq[0]) for sq in data]
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    data = pad_sequence(x, batch_first=True, padding_value=0)   # 用RNN处理变长序列的必要操作
    return data, torch.tensor(y, dtype=torch.float32), data_length
def load_tt(batch_size):
    print('开始加载训练和测试样本集')
    time_start = time.time()
    train_data = MyDataset(xyl_train,yyl_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    test_data = MyDataset(xyl_test,yyl_test)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    time_end = time.time()
    time_sum = time_end - time_start
    print("已加载训练和测试样本集,共用时{}s".format(time_sum))
    return train_loader,test_loader


if __name__ == '__main__':
    stopwords_path = "./data/stopwords.txt"
    pos_path = "./data/neg新.xlsx"
    neg_path='./data/pos新.xlsx'
    split=0.2
    model = gensim.models.KeyedVectors.load_word2vec_format('./data/sgns.renmin.word', encoding="utf-8")
    model0 = model.save('./models/word2vec.pkl')
    xyl_train,xyl_test,yyl_train,yyl_test=load_ylk(pos_path,neg_path,stopwords_path,split)
    train_loader,test_loader=load_tt(32)
    for i, batch in enumerate(train_loader):#batch[ 172,  517,  164, 1000...],这里i就是data_list长度除以batch_size
        print(i,batch)












def processingt(data):
    data['words']=[jieba.lcut(line) for line in data['words']]
    x1=[]
    for x in data['words']:
        x0=[]
        for word in x:
            if word not in stopwords:
                x0.append(word)
        x1.append(x0)
    for x in range(0,len(data['words'])):
        data['words'].iloc[x]=x1[x]
    return data