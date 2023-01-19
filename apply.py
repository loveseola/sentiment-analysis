import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import datetime
import math
from yqfx_ml.textrank import key_phrases

zhuewang=merged
outdir='D:\\wikiword2vec\\baiduAPI\\genxin'
address=os.path.join(outdir,'yuce.csv')
mergedgd2_baidu=pd.read_csv(address)#应该会多一部分索引
mergedgd2_baidu.columns=zhuewang.columns
mergedgd2_baidu=mergedgd2_baidu.drop_duplicates(subset=['level_0','index','时间'], keep='first')#去除重复的索引
zhuewang_=zhuewang[~zhuewang['时间'].isin(mergedgd2_baidu['时间'].unique().tolist())]
zhewang_baidu=zhuewang_.append(mergedgd2_baidu)
#板块一：正负情感比例
zhuewangrq = zhewang_baidu.groupby([zhewang_baidu['时间'],zhewang_baidu['综合情感值']]).agg({'内容': 'count'})
x = zhewang_baidu.groupby(zhewang_baidu['时间']).agg({'内容': 'count'})
y = zhuewangrq.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index() #算正负文章百分比
z =x.groupby(level=0).apply(lambda x: x.sum()).reset_index()#每日总文章
list = ['x', 1, 'y', 2, 'z', 3]
rq=z['时间'].tolist()
zwz =z.iloc[:,-1].tolist()
dic = dict(zip(rq,zwz))
y['总文章'] = y['时间'].map(dic)
y = y.pivot(columns ='综合情感值',index ='时间',values ='内容')
y =y.fillna(0)
yx = pd.merge(y,x,on = y.index)
yx.columns =['时间','负面比例','正面比例','总文章']
yx['正负情感比例']=0
for a in range(0,len(yx['正负情感比例'])):
  yx['正负情感比例'].iloc[a]=str(round(yx['正面比例'].iloc[a]))+':'+str(round(yx['负面比例'].iloc[a]))
result = yx[['正负情感比例','时间']]
#板块二：情感值
zhewang_baidu =zhewang_baidu.dropna(subset =['标题', '内容'])
zhewang_baidu =zhewang_baidu.drop_duplicates(subset=['内容'], keep='first')
zhuewangrq = zhewang_baidu.groupby([zhewang_baidu['时间'],zhewang_baidu['综合情感值']]).count().reset_index()
zhuewangrq =zhuewangrq.set_index('时间')
bi0=[]
for x in zhuewangrq.index.unique():
    n = zhuewangrq.loc[zhuewangrq.index==x].groupby('综合情感值').sum()
    if 1 not in n.index:
        npos =0
    else:
        npos = n.loc[n.index==1]['内容'].values
    if 0 not in n.index:
        nneg =0
    else:
        nneg = n.loc[n.index==0]['内容'].values
    npos_neg = round(math.log((1+npos)/(1+nneg)),2) #情感值计算公式
    bi0.append([x,npos_neg])
bi0 = pd.DataFrame(bi0,columns=['日期','看涨指数情感值'])
bi0 = bi0.set_index('日期')
result = bi0.reset_index()
#板块三：关键短语和关键词
result=key_phrases('./yqfx_ml/data/test.xlsx',2,20,1,'./yqfx_ml/data/stopwords.txt')
result=key_words('./yqfx_ml/data/test.xlsx',2,20,1,'./yqfx_ml/data/stopwords.txt')