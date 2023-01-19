import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from textrank4zh import TextRank4Sentence,TextRank4Keyword
#关键句提取
def key_sentence(pre_data_path,key_num):
    '''
    :param pre_data_path: 待预测的dataframe
    :param key_num: 关键句数量
    :return: 新增关键句列的dataframe
    '''
    tr4s = TextRank4Sentence()
    pre_data_path['关键句'] = 0
    # 先取每篇文章的关键句
    pre_data_path_2 = []
    for article in pre_data_path['内容']:
        x = re.sub('\r', '', str(article))
        # x=re.sub(re.compile(r'[^\u4E00-\u9FA5|\s\w]'), '', x)#非中文、英文、表情的都干掉（中文unicode编码范围：[0x4E00,0x9FA5]）
        x = re.sub(re.compile(r'[a-zA-z]+'), '', x)  # 去除英文
        x = x.strip('\n')
        tr4s.analyze(text=x, lower=True, source='all_filters')
        pre_data_path_0 = []
        for item in tr4s.get_key_sentences(num=key_num):
            pre_data_path_0.append([item.sentence, item.index])  # index是语句在文本中位置，weight是权重
            pre_data_path0_ = pd.DataFrame(pre_data_path_0, columns=['关键句', '索引位置']).sort_values('索引位置')
        pre_data_path_2.append('。'.join(pre_data_path0_['关键句'].tolist()))
    for x in range(0, len(pre_data_path)):
        pre_data_path['关键句'].iloc[x] = str(pre_data_path_2[x])
    return pre_data_path

def key_phrases(pre_data_path,window,key_num,occur_num,stopwords_path):
    '''
    :param pre_data_path: 待预测的dataframe
    :param window: 窗口大小，项目里默认为2
    :param key_num: 关键短语个数，默认为20
    :param stopwords_path:停用词词典

    :return:新增关键短语的dataframe
    '''
    tr4w = TextRank4Keyword(stop_words_file = stopwords_path)
    pre_data_path['关键短语']=0
    pre_data_path_2=[]
    for article in pre_data_path['关键句']:
        x =re.sub('\r','',str(article))
        x= x.strip('\n')
        tr4w.analyze(text=x, lower=True, window=window)
        pre_data_path_0 = []
        for phrase in tr4w.get_keyphrases(keywords_num=key_num, min_occur_num=occur_num):
            pre_data_path_0.append(phrase)
        pre_data_path_2.append(pre_data_path_0)
    for x in range(0,len(pre_data_path)):
        pre_data_path['关键短语'].iloc[x]=str(pre_data_path_2[x])
    return pre_data_path
def key_words(pre_data_path,window,key_num,word_len,stopwords_path):
    '''
    :param pre_data_path: 待预测的dataframe
    :param window: 窗口大小，项目里默认为2
    :param key_num: 关键词个数，默认为20
    :param word_len:词最小长度
    :return:新增关键词的dataframe
    '''
    tr4w = TextRank4Keyword(stop_words_file=stopwords_path)
    pre_data_path['关键词'] = 0
    pre_data_path_2=[]
    for article in pre_data_path['关键句']:
        x =re.sub('\r','',str(article))
        x= x.strip('\n')
        tr4w.analyze(text=x, lower=True, window=window)
        pre_data_path_0 = []
        for item in tr4w.get_keywords(key_num, word_min_len=word_len):
            pre_data_path_0.append(item.word)
        pre_data_path_2.append(pre_data_path_0)
    for x in range(0,len(pre_data_path)):
        pre_data_path['关键词'].iloc[x]=str(pre_data_path_2[x])
    return pre_data_path