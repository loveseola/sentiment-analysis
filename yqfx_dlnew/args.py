# -*- coding: UTF-8 -*-
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import argparse, torch

def lstm_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoches', type=int, default=10, help='num_epoches')
    parser.add_argument('--embedding_dim', type=int, default=300, help='embed dimension')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden size')
    parser.add_argument('--num_layers', type=int, default=5, help='num layers')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--split',type=float,default= 0.2,help='split')
    parser.add_argument('--num_classes', type=float, default=2, help='classes')
    parser.add_argument('--df',type=str,default='./data/test.xlsx',help='yqfxdata')
    parser.add_argument('--neg_data_path',type=str,default='./data/pos新.xlsx',help='yqfxneg')
    parser.add_argument('--pos_data_path',type=str,default="./data/neg新.xlsx",help='yqfxpos')
    parser.add_argument('--stopwords_path', type=str, default="./data/stopwords.txt", help='stopwords')
    parser.add_argument('--modelname', type=str, default="BILSTM", help='modelname')
    args = parser.parse_args()
    return args
