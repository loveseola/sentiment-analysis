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
import traceback

#0ne:语料库预处理
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
    num_tokens = [len(tokens) for tokens in pos_and_neg]
    num_tokens = np.array(num_tokens)
    np.mean(num_tokens)
    max_tokens = int(np.mean(num_tokens) + 2 * np.std(num_tokens))
    return xyl_train,xyl_test,yyl_train,yyl_test,max_tokens,pos_and_neg
#two:建立语料库词汇表及文本索引
def build_word2id(file,pos_and_neg):
    '''
    :param file:建立语料库词汇表
    :param pos_and_neg:语料库
    :return: 将可迭代文本建立到数字索引的映射亦可参考
    https://github.com/sariel-black/taptap_emotion_analyse/blob/master/taptap%E8%AF%84%E8%AE%BA%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90/tap%20emotion%20analyse%20-v2.ipynb 使用自建词向量模型
    '''
    #建立语料库的词汇表
    word2id = {"_PAD_": 0}
    for sentence in pos_and_neg:
        for word in sentence:
            if word not in word2id.keys():
                word2id[word]=len(word2id) #词对应的位置
    with open(file,"w",encoding="utf-8") as f:
        for w in word2id:
            f.write(w+"\t")
            f.write(str(word2id[w]))
            f.write("\n")
    return word2id
def text_to_array(w2id, seq_lenth, x,label):
    '''
    :param w2id:已建立的语料库词汇表
    :param seq_lenth:控制的长度，详见load_ylk函数
    :param x: 可迭代文本
    :param label:可迭代文本标签
    :return:文本数据集转换数字索引，shape为len(sa), seq_lenth)，标签shape为(len(sa),1)
    '''
    label_array = []
    i = 0
    sa = []
    # 获取句子个数
    for sentence in x:
        new_s=[w2id.get(word, 0) for word in sentence]# 获取键值，不存在返回0。单词转索引数字
        sa.append(new_s)
    sentences_array = np.zeros(shape=(len(sa), seq_lenth))  # 行：句子个数 列：句子长度
    for sentence in x:
        new_sen=[w2id.get(word, 0) for word in sentence] # 单词转索引数字,不存在则为0
        new_sen_np = np.array(new_sen).reshape(1, -1)#列表转化为数组,转化成一行，shape为（1，len(new_sen))
        # 补齐每个句子长度，多余补零，少了就直接赋值,0填在前面。
        if np.size(new_sen_np, 1) < seq_lenth:
            sentences_array[i, seq_lenth - np.size(new_sen_np, 1) :] = new_sen_np[0, :] #array()
        else:
            sentences_array[i, 0:seq_lenth] = new_sen_np[0, 0:seq_lenth]
        i = i + 1
        label_array=label# 标签，np.array([a])增加一个维度
    return np.array(sentences_array), np.array([label_array]).T

# three：数据集
class Data_set(Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        if Label is not None:  # 考虑对测试集的使用
            self.Label = Label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        if self.Label is not None:
            data = torch.from_numpy(self.Data[index])#转化成tensor
            label = torch.from_numpy(self.Label[index])
            return data, label
        else:
            data = torch.from_numpy(self.Data[index])
            return data
def load_tt(batch_size,train_loader,test_loader):
    '''
    :param batch_size:
    :param train_loader: 训练数据集
    :param test_loader: 测试训练集
    :return: 训练和测试样本集
    '''
    print('开始加载训练和测试样本集')
    time_start = time.time()
    train_dataloader = DataLoader(train_loader, batch_size=batch_size, shuffle=True,
                                  num_workers=0)  # 相当于做了压缩 每多少个数打包在一起，同时
    test_dataloader = DataLoader(test_loader, batch_size=batch_size, shuffle=True, num_workers=0)
    time_end = time.time()
    time_sum = time_end - time_start
    print("已加载训练和测试样本集,共用时{}s".format(time_sum))
    return train_dataloader,test_dataloader
def build_word2vec(word2id,model,save_to_path=None):
    '''
    :param word2id:已建立的语料库词汇表
    :param model:词向量
    :param save_to_path:
    :return:初始化的词汇向量矩阵
    '''
    n_words=max(word2id.values())+1 #词汇量
    word_vecs=np.array(np.random.uniform(-1.0,1.0,[n_words,model.vector_size]))#均匀分布[low,high)中随机采样
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]]=model[word]
        except KeyError as e:
            traceback.print_exc(file = open('./data/keyerrorlog_abc.txt','a'))
    if save_to_path:
        with open(save_to_path,"w",encoding="utf-8") as f:
            for vec in word_vecs:
                vec=[str(w) for w in vec]
                f.write(" ".join(vec))
                f.write("\n")
    return word_vecs
def processingt(data_pre_path,stopwords_path):
    '''
    :param data_pre_path: 待预测的文本,已进行数据清洗
    :param stopwords_path: 停用词词典
    :return: 预处理的待预测文本
    '''
    stopwords = pd.read_csv(stopwords_path, header=None, quoting=csv.QUOTE_NONE, delimiter="\t")
    stopwords = stopwords[0].tolist()
    data = pd.read_excel(data_pre_path)
    data=data.dropna(subset=['words'])
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
def text_to_array_n(w2id, seq_lenth, x):
    '''
    :param w2id:已建立的语料库词汇表
    :param seq_lenth:控制的长度，详见load_ylk函数
    :param x: 可迭代文本
    :return:文本数据集转换数字索引，shape为len(sa), seq_lenth)，标签shape为(len(sa),1)
    '''
    i = 0
    sa = []
    # 获取句子个数
    for sentence in x:
        new_s=[w2id.get(word, 0) for word in sentence]# 获取键值，不存在返回0。单词转索引数字
        sa.append(new_s)
    sentences_array = np.zeros(shape=(len(sa), seq_lenth))  # 行：句子个数 列：句子长度
    for sentence in x:
        new_sen=[w2id.get(word, 0) for word in sentence] # 单词转索引数字,不存在则为0
        new_sen_np = np.array(new_sen).reshape(1, -1)
        # 补齐每个句子长度，多余补零，少了就直接赋值,0填在前面。
        if np.size(new_sen_np, 1) < seq_lenth:
            sentences_array[i, seq_lenth - np.size(new_sen_np, 1) :] = new_sen_np[0, :]
        else:
            sentences_array[i, 0:seq_lenth] = new_sen_np[0, 0:seq_lenth]
        i = i + 1
    return np.array(sentences_array)
'''
if __name__ == '__main__':
    stopwords_path = "./data/stopwords.txt"
    pos_data_path = "./data/neg新.xlsx"
    neg_data_path='./data/pos新.xlsx'
    split=0.2
    batch_size=32
    model = gensim.models.KeyedVectors.load_word2vec_format('./data/sgns.renmin.word', encoding="utf-8")
    model0 = model.save('./models/word2vec.pkl')
    xyl_train,xyl_test,yyl_train,yyl_test,max_tokens,pos_and_neg=load_ylk(pos_data_path,neg_data_path,stopwords_path,split)
    word2id = build_word2id('./data/word2id.txt',pos_and_neg)
    train_array, train_label = text_to_array(word2id, max_tokens, xyl_train, yyl_train)  # shape:(len(train), seq_len)
    test_array, test_label = text_to_array(word2id, max_tokens, xyl_test, yyl_test)
    train_loader = Data_set(train_array, train_label)
    test_loader = Data_set(test_array, test_label)
    train_dataloader,test_dataloader = load_tt(batch_size,train_loader,test_loader)
    for i, batch in enumerate(train_dataloader):#batch[ 172,  517,  164, 1000...],这里i就是data_list长度除以batch_size
        print(i,batch)
'''



