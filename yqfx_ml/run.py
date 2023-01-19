import warnings
warnings.filterwarnings("ignore")
"""
author:Liao Xuechun
date:2023-01-19
describe: ml sentiment analysis 
"""
import os
import re
import jieba
import pandas as pd
import numpy as np
import csv
import openpyxl
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
import gensim
import joblib
import tensorflow as tf
from keras.models import load_model
from datasets import load_ylk,word_to_dict,text_to_index_array,embedding_weights,buildVecs,getWordVecs
from args import args,keras_lstm,svm,dt
from model import svm_train,NB_train,decision_tree,random_forest,gbdt_classifier,xgboost,train_lstm




if __name__=='__main__':
    args=args()
    print("开始加载词向量")
    model = gensim.models.KeyedVectors.load_word2vec_format('./data/sgns.renmin.word', encoding="utf-8")
    xyl_train,xyl_test,yyl_train,yyl_test,max_tokens,pos_and_neg=load_ylk(args.pos_data_path,args.neg_data_path,args.stopwords_path,args.split)
    train_vec=buildVecs(xyl_train,model)
    train_vec= np.array(train_vec)
    test_vec=buildVecs(xyl_test,model)
    test_vec = np.array(test_vec)
    if args.modelname=='svm':
        svm_args=svm()
        print("开始训练")
        svm=svm_train(train_vec, yyl_train, test_vec, yyl_test, svm_args.cf)
    if args.modelname=='nb':
        print("开始训练")
        NB=NB_train(train_vec,yyl_train,test_vec,yyl_test)
    if args.modelname=='dt':
        dt_args = dt()
        print("开始训练")
        dt=decision_tree(train_vec,yyl_train,test_vec,yyl_test,dt_args.depth,dt_args.samples_split)
    if args.modelname=='rf':
        rf_args = rf()
        print("开始训练")
        rf=random_forest(train_vec,yyl_train,test_vec,yyl_test,rf_args.estimator,rf_args.sample)
    if args.modelname=='gbdt':
        gbdt_args = gbdt()
        print("开始训练")
        gbdt=gbdt_classifier(train_vec,yyl_train,test_vec,yyl_test,gbdt_args.estimator,gbdt_args.lr,gbdt_args.depth)
    if args.modelname=='xgboost':
        xgboost_args = gbdt()
        print("开始训练")
        xgboost=xgboost(train_vec,yyl_train,test_vec,yyl_test,xgboost_args.depth,xgboost_args.estimator,xgboost_args.sample_b)
    #keras封装；
    if args.modelname=='keras_lstm':
        model_args=keras_lstm()
        word_to_vec1, word_to_index1 = word_to_dict(pos_and_neg, model)
        X_train,y_train = text_to_index_array(word_to_index1, xyl_train,yyl_train)  # 全部转化成数字矩阵
        X_test,y_test = text_to_index_array(word_to_index1, xyl_test,yyl_test)
        print("填充词向量矩阵")
        embedding_weights, vocab = embedding_weights(word_to_index1, word_to_vec1,model)
    # 截断
        X_train0 = tf.keras.utils.pad_sequences(X_train, maxlen=int(max_tokens), padding='pre', truncating='pre')
        X_test0 = tf.keras.utils.pad_sequences(X_test, maxlen=int(max_tokens), padding='pre', truncating='pre')
        print("开始训练")
        train_lstm(vocab,model_args.vocab_dim, embedding_weights,max_tokens, X_train0, y_train, X_test0, y_test,model_args.num_1,model_args.num_2,model_args.batch_size,model_args.n_epoch)
        modelx = load_model('./models/emotion_model_LSTM.h5')
        loss, acc = modelx.evaluate(X_test0, y_test, batch_size=model_args.batch_size)
        # test_p=modelx.predict(X_test0)
        # s0=sum(test_p==y_test)/len(y_test)
        lstmresult = pd.DataFrame([acc], index=['krlstmacc'])
        print(lstmresult)
