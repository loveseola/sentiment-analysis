import warnings
warnings.filterwarnings("ignore")
"""
author:Liao Xuechun
date:2023-01-19
describe: ml sentiment analysis 
"""
import numpy as np
import pandas as pd
import csv
import jieba
import gensim
import joblib
from datasets import getWordVecs,processingt
from textrank import key_sentence
from args import args

def predict(pre_data_path,modelselect,stop_word_path):
    #去停用词
    stopwords = pd.read_csv(stop_word_path,header=None,quoting = csv.QUOTE_NONE,delimiter="\t")
    stopwords = stopwords[0].tolist()
    model = gensim.models.KeyedVectors.load_word2vec_format('./data/sgns.renmin.word', encoding="utf-8")
    article_sentiment =[]
    article_sentiment_ =[]
    article_sentiment__ =[]
    sentence_sentiment =[]
    sentence_sentiment_ =[]
    sentence_sentiment__ =[]
    for article in pre_data_path['内容']:
        article = ''.join(re.sub(re.compile(r'[^\u4e00-\u9fa5]'), '', str(article)))
        jieba.add_word("能繁母猪")
        jieba.add_word("美盘")
        words = jieba.lcut(str(article))
        for word in words:
            if word in stopwords and len(word)>=1:
                words.remove(word)
        x = getWordVecs(words,model)
        if len(x) >0:
            vecsArray = sum(np.array(x))/len(x) # mean
        vecsArray = vecsArray.reshape(1,300)
        #预测
        clf = joblib.load(modelselect)
        result = clf.predict(vecsArray)
        result_p=clf.predict_proba(vecsArray)
        article_sentiment.append(result[0])
        article_sentiment_.append(round(result_p[0][0],2))
        article_sentiment__.append(round(result_p[0][1],2))
    for sentence in pre_data_path['关键句']:
        sentence = ''.join(re.sub(re.compile(r'[^\u4e00-\u9fa5]'), '', str(sentence)))
        jieba.add_word("能繁母猪")
        jieba.add_word("美盘")
        words_ = jieba.lcut(str(sentence))
        for word_ in words_:
            if word_ in stopwords and len(word_)>=1:
                words_.remove(word_)
        y = getWordVecs(words_,model)
        if len(y) >0:
            vecsArray_ = sum(np.array(y))/len(y) # mean
        vecsArray_ = vecsArray_.reshape(1,300)
        clf = joblib.load(modelselect)
        result_ = clf.predict(vecsArray_)
        result_p_=clf.predict_proba(vecsArray_)
        sentence_sentiment.append(result_[0])
        sentence_sentiment_.append(round(result_p_[0][0],2))
        sentence_sentiment__.append(round(result_p_[0][1],2))
        #将情绪结果合并到原数据文件中
    merged = pd.concat([pd.Series(article_sentiment, name='文章情绪'),pd.Series(article_sentiment_, name='文章负面情绪概率'),pd.Series(article_sentiment__, name='文章正面情绪概率'),pd.Series(sentence_sentiment, name='关键句情绪'),pd.Series(sentence_sentiment_, name='关键句负面情绪概率'),pd.Series(sentence_sentiment__, name='关键句正面情绪概率')], axis=1)
    return merged
if __name__ =='__main__':
    #one:用自己训练的模型预测
    args=args()
    data_pre_path='./data/test.xlsx'
    stopwords_path='./data/stopwords.txt'
    data=processingt(data_pre_path,stopwords_path)
    df=key_sentence(data,3)
    mergedgd1 = predict(df,'./models/'+args.modelname+'_model.pkl',stopwords_path)
    mergedgd=pd.concat([df,mergedgd1],axis=1)
    print(mergedgd)
    #two:实际部署的时候加上了百度情感值得到了综合情感值
    '''
    #百度情感值
    ak = '百度智能云相关ak'
    sk = '百度智能云相关sk'
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={}&client_secret={}'.format(ak,sk)
    response = requests.post(host)
    token = response.json().get('access_token')
    def main(token,textinput):
        url = "https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?access_token={}&charset=UTF-8".format(token) #获取URL
        payload = json.dumps({'text':textinput})
        headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }   
        if len(textinput)<=100:
            response = requests.request("POST", url, headers=headers, data=payload) #发送请求
            x=response.text
            time.sleep(3)
        elif len(textinput) >100:   
            response = requests.request("POST", url, headers=headers, data=payload) #发送请求
            x=response.text
            time.sleep(5)        
        return json.loads(x)
    #分今日迭代和过往
    df1=df[df['时间']<=(datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')].reset_index()
    df2=df[df['时间']== (datetime.datetime.now()).strftime('%Y-%m-%d')].reset_index()
    df2['百度API']=df2['关键句'].apply(lambda x:main(token,str(x)))#['items'][0]['sentiment']
    df2['百度情感值']=df2['百度API'].apply(lambda x:x['items'][0]['sentiment'])#
    df2['百度正面情感概率']=df2['百度API'].apply(lambda x:x['items'][0]['positive_prob'])#
    df2['百度负面情感概率']=df2['百度API'].apply(lambda x:x['items'][0]['negative_prob'])#
    mergedgd1 = svm_predict(df1, 'D:/wikiword2vec/svm_model.pkl')  # 旧的
    mergedgd1_ = pd.concat([df1, mergedgd1], axis=1)
    mergedgd1_['综合情感值'] = mergedgd1_['关键句情绪'].apply(lambda x: x)
    mergedgd2 = svm_predict(df2, 'D:/wikiword2vec/svm_model_daynew.pkl')  # 当日迭代的
    mergedgd2_ = pd.concat([df2, mergedgd2], axis=1)
    mergedgd2_['综合情感值'] = mergedgd2_['关键句情绪'] * 0.4 + mergedgd2_['百度情感值'] * 0.6
    mergedgd2_['综合情感值'] = mergedgd2_['综合情感值'].apply(lambda x: 1 if x >= 0.5 else 0)
    outdir = 'F:\\E\\lxc\\nlp\\yqfx\\yqfxzl\\yqfx_ml'
    paramDir = 'yuce'
    save0 = os.path.join(outdir, 'gengxin')
    if not os.path.exists(save0):
        os.makedirs(save0)
        address = os.path.join(save0, paramDir + '.csv')
        megedgd2_.columns = mergedgd1_.columns
        mergedgd2_.to_csv(address, encoding='utf-8')  # 拼接路径名
    if os.path.exists(save0):
        address = os.path.join(save0, paramDir + '.csv')
        mergedgd2_.to_csv(address, mode='a', encoding='utf-8', index=False, header=False)  # 追加
    # mergedgd2_baidu=pd.read_csv(address)#应该会多一部分索引
    mergedgd = mergedgd1_.append(mergedgd2_)
    # 对原料信息的实际情绪进行修改
    mergedgd_yl = mergedgd[mergedgd['标题'].str.contains('玉米|豆|小麦')]
    mergedgd_yl['文章情绪'] = mergedgd_yl['文章情绪'].apply(lambda x: x - 1 if x == 1 else x + 1)
    mergedgd_yl['关键句情绪'] = mergedgd_yl['关键句情绪'].apply(lambda x: x - 1 if x == 1 else x + 1)
    # mergedgd_yl['正面情绪概率']=mergedgd_yl['正面情绪概率'].apply(lambda x:round(1-x,2))
    # mergedgd_yl['负面情绪概率']=mergedgd_yl['负面情绪概率'].apply(lambda x:round(1-x,2))
    merged = mergedgd_yl.append(mergedgd[~mergedgd['标题'].str.contains('玉米|豆|小麦')])
'''


