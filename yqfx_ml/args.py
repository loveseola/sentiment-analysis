# -*- coding: UTF-8 -*-
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse
def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=float, default=0.2, help='split')
    parser.add_argument('--modelname', type=str, default="svm", help='modelname')
    parser.add_argument('--df',type=str,default='分析数据.xls',help='yqfxdata')
    parser.add_argument('--neg_data_path',type=str,default='./data/pos新.xlsx',help='yqfxneg')
    parser.add_argument('--pos_data_path',type=str,default="./data/neg新.xlsx",help='yqfxpos')
    parser.add_argument('--stopwords_path', type=str, default="./data/stopwords.txt", help='stopwords')
    args = parser.parse_args()
    return args
def keras_lstm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_dim', type=int, default=300, help='vocab dimension')
    parser.add_argument('--num_1', type=int, default=64, help='hidden size1')
    parser.add_argument('--num_2', type=int, default=64, help='hidden size2')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--n_epoch',type=float,default= 10,help='n_epoch')
    keras_lstm_args = parser.parse_args()
    return keras_lstm_args
def svm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cf', type=int, default=2, help='chengfa')
    svm_args = parser.parse_args()
    return svm_args

def dt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=str, default="keras_lstm", help='depth')
    parser.add_argument('--samples_split',type=str,default='分析数据.xls',help='samples_split')
    dt_args = parser.parse_args()
    return dt_args
def rf():
    parser = argparse.ArgumentParser()
    parser.add_argument('--estimator', type=str, default="keras_lstm", help='estimator')
    parser.add_argument('--sample',type=str,default='分析数据.xls',help='sample')
    rf_args = parser.parse_args()
    return rf_args
def gbdt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--estimator', type=str, default="keras_lstm", help='estimator')
    parser.add_argument('--depth',type=str,default='分析数据.xls',help='depth')
    parser.add_argument('--lr', type=str, default='分析数据.xls', help='lr')
    gbdt_args = parser.parse_args()
    return gbdt_args

def xgboost():
    parser = argparse.ArgumentParser()
    parser.add_argument('--estimator', type=str, default="keras_lstm", help='estimator')
    parser.add_argument('--depth', type=str, default='分析数据.xls', help='depth')
    parser.add_argument('--sample_b', type=str, default='分析数据.xls', help='sample_b')
    xgboost_args = parser.parse_args()
    return xgboost_args