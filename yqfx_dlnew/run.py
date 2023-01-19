import warnings
warnings.filterwarnings("ignore")
"""
author:Liao Xuechun
date:2023-01-19
describe: dl sentiment analysis 
"""
import re
import jieba
import pandas as pd
import numpy as np
import csv
import openpyxl
import os
from sklearn.model_selection import train_test_split
from gensim.models import word2vec
import gensim
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import time
import traceback
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, recall_score
import tqdm
from datasets import load_ylk,load_tt,build_word2vec,build_word2id,text_to_array,Data_set,processingt,text_to_array_n
from models import BILSTMModel,BiLSTM_Attention
from args import lstm_parser
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def test_accuary(test_dataloader,model):
    '''
    :param test_dataloader: 测试集
    :param model: 模型
    :return: 测试集准确率
    '''
    with torch.no_grad():
        correct = 0
        total = 0
        for k, data_test in enumerate(test_dataloader, 0):
            input_test, target_test = data_test[0], data_test[1]
            input_test = input_test.type(torch.LongTensor)
            target_test = target_test.type(torch.LongTensor)
            target_test = target_test.squeeze(1)
            input_test = input_test.to(device)
            target_test = target_test.to(device)
            output_test = model(input_test)
            _, pred_test = torch.max(output_test, 1)
            total += target_test.size(0)#个数

            correct += (pred_test == target_test).sum().item()
            F1 = f1_score(target_test.cpu(), pred_test.cpu(), average="weighted")
            Recall = recall_score(target_test.cpu(), pred_test.cpu(), average="micro")
            CM = confusion_matrix(target_test.cpu(), pred_test.cpu())
        print(
            "\nTest accuracy : {:.3f}%, F1_score: {:.3f}%, Recall: {:.3f}%, Confusion_matrix: {}".format(
                100 * correct / total, 100 * F1, 100 * Recall, CM
            )
        )
    return correct / total
def train(train_dataloader,test_dataloader,model,modelname,lr,num_epochs):
    '''
    :param train_dataloader: 训练集
    :param test_dataloader: 测试集
    :param model: 模型
    :param modelname: 模型名称
    :param lr: 学习率
    :param num_epochs: 迭代次数
    :return: 迭代准确率
    '''
    print('开始训练')
    time_start = time.time()
    model.train()
    model=model.to(device)
    loss=nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    acc_=[]
    for epoch in range(num_epochs):
        train_loss,correct,total= 0.0,0.0,0
        train_dataloader = tqdm.tqdm(train_dataloader)
        for i,data_ in enumerate(train_dataloader):
            input_, target = data_[0], data_[1]
            #print(input_.shape)
            #变量移动到GPU
            input_ = input_.type(torch.LongTensor).to(device)#必须转化为long，才能nn.embedding
            target = target.type(torch.LongTensor).to(device)
            #训练
            optimizer.zero_grad()  #梯度清零
            output = model(input_)   #模型output
            target = target.squeeze(1)#实际目标label:target, shape:[num_samples, 1]=>[num_samples]
            l = loss(output, target)#loss计算
            l.backward()# 反向传播，计算梯度
            optimizer.step()# 梯度更新
            train_loss += l.item()#损失记录
            _, predicted = torch.max(output, 1)#按维度dim 返回最大值
            total+=target.size(0)
            correct+=(predicted == target).sum().item() #等价于(predicted.argmax(dim=1) == target).sum().item()
            F1 = f1_score(target.cpu(), predicted.cpu(), average="weighted")
            Recall = recall_score(target.cpu(), predicted.cpu(), average="micro")
            postfix = {"train_loss: {:.5f},train_acc:{:.3f}%"
                ",F1: {:.3f}%,Recall:{:.3f}%".format(
                    train_loss / (i + 1), 100 * correct / total, 100 * F1, 100 * Recall
                )
            }
            train_dataloader.set_postfix(log=postfix)#打印日志
        acc = test_accuary(test_dataloader,model)
        acc_.append(acc)
        model_path = "./models/"+modelname+"_{}.model".format(epoch+1)
        torch.save(model, model_path)
        print("saved model: ", model_path)
    time_end=time.time()
    time_sum=time_end-time_start
    print("训练共用时{}s".format(time_sum))
    if not os.path.exists("./results/"+modelname):
        os.makedirs("./results/"+modelname)
    np.savetxt("./results/"+modelname+"/pl_modelselect.txt",acc_)#模型选择
    return acc_

def pre(word2id, model, modelname,seq_lenth, path):
    '''
    :param word2id: 已建立的语料库词汇表
    :param model: 训练的最优模型
    :param modelname: 模型名称
    :param seq_lenth: seq_lenth:控制的长度，详见load_ylk函数
    :param path: 预处理完后的待预测文本
    :return: 预测结果
    '''
    print('开始预测')
    with torch.no_grad():
        input_array = text_to_array_n(word2id, seq_lenth, path['words'])
        sen_p = torch.from_numpy(input_array)
        sen_p = sen_p.type(torch.LongTensor)
        sen_p=sen_p.to(device)
        output_p = model(sen_p)
        _, pred = torch.max(output_p, 1)#输出最大值所在维度
        path['文章情绪']=0
        for i in range(pred.size(0)):
            path['文章情绪'].iloc[i]=pred[i].item()
    path.to_csv("./results/"+modelname+"/plpre-Result.csv")
    print('预测完毕')
    return path
if __name__=='__main__':
    time_start=time.time()
    #读取参数
    args = lstm_parser()
    #词向量
    print("开始加载词向量")
    model = gensim.models.KeyedVectors.load_word2vec_format('./data/sgns.renmin.word', encoding="utf-8")
    xyl_train,xyl_test,yyl_train,yyl_test,max_tokens,pos_and_neg=load_ylk(args.pos_data_path,args.neg_data_path,args.stopwords_path,args.split)
    word2id = build_word2id('./data/word2id.txt',pos_and_neg)
    print('开始准备模型数据')
    train_array, train_label = text_to_array(word2id, max_tokens, xyl_train, yyl_train)  # shape:(len(train), seq_len)
    test_array, test_label = text_to_array(word2id, max_tokens, xyl_test, yyl_test)
    train_loader = Data_set(train_array, train_label)
    test_loader = Data_set(test_array, test_label)
    train_dataloader, test_dataloader = load_tt(args.batch_size, train_loader, test_loader)
    print('模型数据准备完毕')
    w2vec = build_word2vec(word2id, model,'./data/word2vec.txt')  # shape (70278, max_tokens)
    w2vec = torch.from_numpy(w2vec)
    w2vec = w2vec.float()
    BILSTM = BILSTMModel(max(word2id.values())+1, args.embedding_dim, w2vec, args.hidden_dim, args.num_layers,args.num_classes)
    BILSTMAttention= BiLSTM_Attention(max(word2id.values()) + 1, args.embedding_dim, w2vec, args.hidden_dim, args.num_layers, args.num_classes)
    if args.modelname=='BILSTM':
        BILSTM_train=train(train_dataloader,test_dataloader,BILSTM,args.modelname,args.learning_rate,args.num_epoches)
        df_=processingt(args.df,args.stopwords_path)
        a = np.loadtxt("./results/"+args.modelname+"/pl_modelselect.txt")
        index=a.tolist().index(max(a.tolist()))
        net = torch.load("./models/"+args.modelname+"_{}.model".format(index+1))
        x = pre(word2id, net, args.modelname,max_tokens, df_)
        print(x.head())
    if args.modelname=='BILSTM_Attention':
        BiLSTM_Attention_train=train(train_dataloader,test_dataloader,BILSTMAttention,args.modelname,args.learning_rate,args.num_epoches)
        df_=processingt(args.df,args.stopwords_path)
        a = np.loadtxt("./results/"+args.modelname+"/pl_modelselect.txt")
        index=a.tolist().index(max(a.tolist()))
        net = torch.load("./models/"+args.modelname+"_{}.model".format(index+1))
        x = pre(word2id, net, args.modelname,max_tokens, df_)
        print(x.head())
    time_end=time.time()
    time_sum=time_end-time_start
    print('程序运行共用时{}s'.format(time_sum))