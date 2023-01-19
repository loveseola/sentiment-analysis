import warnings
warnings.filterwarnings("ignore")
import re
import jieba
import pandas as pd
import numpy as np
import csv
import openpyxl
import traceback
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
import gensim

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
    num_tokens = [len(tokens) for tokens in pos_and_neg]
    num_tokens = np.array(num_tokens)
    np.mean(num_tokens)
    max_tokens = int(np.mean(num_tokens) + 2 * np.std(num_tokens))
    return xyl_train,xyl_test,yyl_train,yyl_test,max_tokens,pos_and_neg
def getWordVecs(wordList,model):
    '''
    :param wordList:每一条分完词后的语料
    :param model: 词向量模型
    :return: 一条语料里所有词的特征词向量
    '''
    vecs = []
    for word in wordList:
        word = word.replace('\n','')
        #print word
        try:
            vecs.append(model[word])
        except KeyError:
            continue
    return np.array(vecs, dtype='float')
def buildVecs(sent,model):
    '''
    :param sent: 训练、测试语料集
    :param model: 词向量模型
    :return: 每一条语料的平均特征词向量
    '''
    fileVecs = []
    for wordList in sent:
        vecs = getWordVecs(wordList,model)
        if len(vecs) >0:
            vecsArray = sum(np.array(vecs))/len(vecs) # mean
            fileVecs.append(vecsArray)
    return fileVecs

def processing_open(data_path,stopwords_path):
    '''

    :param data_path: 语料库,因为项目实际需要这里采用pd.read方式读取
    :param stopwords_path:停用词词典，这里也可以使用open方式读取
    :return:语料库新增分词列
    '''
    stops_word = open(stopwords_path,encoding= 'utf-8').readlines()#读成列表
    stops_word =[line.strip()for line in stops_word]#换行符过滤同样得到列表
    data = pd.read_excel(data_path)
    data.columns = ['content', 'label']
    zw = lambda x: ''.join(re.sub(re.compile(r'[^\u4e00-\u9fa5]'), '', str(x)))
    data['words'] = data['content'].apply(zw)
    x1=[]
    for word in data['words']:
        words = jieba.cut(word,cut_all= False)
        x0=[]
        for words_ in words:
            if words_ in stops_word:
                continue
            x0.append(words_)
        x1.append(x0)
    for x in range(0,len(data['words'])):
        data['words'].iloc[x]=x1[x]
    return data
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

def word_to_dict(X, word_vec_model):
    '''
    :param X:可迭代文本
    :param word_vec_model:词向量模型
    :return:分词所对应词向量
    '''
    total_set = set()
    word_to_vec = dict()
    word_to_index = dict()

    for x in X:
        total_set = set.union(total_set, set(x))#union() 方法返回两个集合的并集
    index = 1
    for i in total_set:
        try:
            word_to_vec[i] = word_vec_model[i]
            word_to_index[i] = index
            index += 1
        except KeyError as e:
            traceback.print_exc(file = open('./data/keyerrorlog_abc.txt','a'))
    return word_to_vec, word_to_index
#出现在索引字典里的词转化为其索引数字if type(p_sen)==np.ndarray:
def text_to_index_array(p_new_index,p_sen,label):
    '''
    :param p_new_index:单词索引
    :param p_sen:可迭代文本
    :param label:标签
    :return:转化后的数字索引
    '''
    new_sentences=[]
    label_array=[]
    for sen in p_sen:
        new_sen=[]
        for word in sen:
            try:
                new_sen.append(p_new_index[word])#单词转化成索引
            except:
                new_sen.append(0)#索引字典里没有的词转成数字0
        new_sentences.append(new_sen)
        label_array=label
    return np.array(new_sentences),np.array(label_array)
def embedding_weights(word_to_index,word_to_vec,model):
    '''
    :param word_to_index:
    :param word_to_vec:
    :return: 词向量矩阵，该处与dlnew下的datasets里的build_word2vec作用相同
    '''
    index_1 = len(word_to_index)
    embedding_weights = np.zeros((index_1 + 1, model.vector_size))
    for w, index in word_to_index.items():
        embedding_weights[index, :] = word_to_vec[w]  # 第一行是0向量
    return embedding_weights,index_1+1
