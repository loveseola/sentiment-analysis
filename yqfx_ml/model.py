import warnings
warnings.filterwarnings("ignore")
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding,LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.models import load_model
import keras
from keras.layers import Bidirectional
import joblib
import pandas as pd
def svm_train(train_vec,yyl_train,test_vec,yyl_test,cf):
    '''
    :param train_vec:训练集特征向量
    :param yyl_train:训练集标签
    :param test_vec:测试集特征向量
    :param yyl_test:测试集标签
    :param cf: 惩罚参数
    :return:svm模型预测准确率
    '''
    clf = SVC(C = cf, probability = True)
    clf.fit(train_vec,yyl_train)
    #持久化保存模型
    joblib.dump(clf,'./models/svm_model.pkl',compress=3)
    svmscore = clf.score(test_vec,yyl_test)
    svmresult = pd.DataFrame([svmscore], index=['svmacc'])
    return svmresult#获取预测结果准确率

def NB_train(train_vec,yyl_train,test_vec,yyl_test):
    '''
    :return:高斯朴素贝叶斯模型准确率
    '''
    gnb = GaussianNB()
    gnb.fit(np.array(train_vec),yyl_train)
    joblib.dump(gnb,'./models/NB_model.pkl')
    gnbscores = gnb.score(np.array(test_vec),yyl_test)
    gnbresult=pd.DataFrame([gnbscores],index=['gnbacc'])
    return gnbresult#获取预测结果准确率

def decision_tree(train_vec,yyl_train,test_vec,yyl_test,depth,samples_split):
    '''
    :param depth: 限制树的最大深度
    :param samples_split: 最小分支训练样本
    :return:决策树模型准确率
    '''
    clf = DecisionTreeClassifier(max_depth=int(depth), min_samples_split=int(samples_split), random_state=0)
    clf.fit(train_vec,yyl_train)
    joblib.dump(clf, './models/dt_model.pkl')
    dtscores = clf.score(test_vec,yyl_test)
    dtresult=pd.DataFrame([dtscores],index=['dtacc'])
    return dtresult#获取预测结果准确率

def random_forest(train_vec,yyl_train,test_vec,yyl_test,estimator,sample):
    '''
    :param estimator:使用决策树个数
    :param sample:有放回抽样的每次抽样样本量
    :return:随机森林模型准确率
    '''
    clf = RandomForestClassifier(n_estimators=int(estimator), max_samples = int(sample), bootstrap = True, max_features = 'auto', oob_score=True,random_state=0)
    clf.fit(train_vec,yyl_train)
    joblib.dump(clf, './models/ranf_model.pkl')
    ranfscores = clf.score(test_vec,yyl_test)
    ranfresult=pd.DataFrame([ranfscores],index=['ranfacc'])
    return ranfresult#获取预测结果准确率

def gbdt_classifier(train_vec,yyl_train,test_vec,yyl_test,estimator,lr,depth):
    '''
    :param estimator：基学习器和决策树数量
    :param lr：学习率
    :param depth：限制树的最大深度
    :return:GBDT模型准确率
    '''
    clf = GradientBoostingClassifier(n_estimators=int(estimator), learning_rate=int(lr), max_depth=int(depth), random_state=0)
    clf.fit(train_vec,yyl_train)
    joblib.dump(clf, './models/gbdt_model.pkl')
    gbdtscores = clf.score(test_vec,yyl_test)
    gbdtresult=pd.DataFrame([gbdtscores],index=['gbdtacc'])
    return gbdtresult#获取预测结果准确率
def xgboost(train_vec,yyl_train,test_vec,yyl_test,depth,estimator,sample_b):
    '''
    :param depth:限制树的最大深度
    :param estimator:基学习器和决策树数量
    :param sample_b:
    :return:构建弱学习器时，对特征随机采样的比例，默认值为1
    '''
    XGB = XGBClassifier(max_depth=int(depth),n_estimators=int(estimator),objective='multi:softmax',num_class=2,booster='gbtree',colsample_bytree=float(sample_b),random_state=0)
    XGB.fit(train_vec,yyl_train)
    y_pred = XGB.predict(test_vec)
    accuracy = accuracy_score(yyl_test, y_pred)
    gbdtresult=pd.DataFrame([accuracy],index=['xgbtacc'])
    return gbdtresult

def train_lstm(p_n_symbols,vocab_dim,p_embedding_weights, max_tokens,p_X_train, p_y_train, p_X_test, p_y_test,num_1,num_2,batch_size,n_epoch):
    '''
    :param p_n_symbols:字典长度
    :param vocab_dim:向量维度
    :param p_embedding_weights:词向量矩阵
    :param max_tokens:pading的长度
    :param p_X_train:训练集文本
    :param p_y_train:训练集标签
    :param p_X_test:测试集文本
    :param p_y_test:测试集标签
    :param num_1:第一层lstm单元个数
    :param num_2:第二层lstm单元个数
    :param batch_size:batch_size
    :param n_epoch:迭代次数
    :return:
    '''
    model = Sequential()
    model.add(Embedding(output_dim=vocab_dim,  # 输出向量维度
                            input_dim=p_n_symbols,  # 字典长度
                            mask_zero=True,  # 使我们填补的0值在后续训练中不产生影响（屏蔽0值）
                            weights=[p_embedding_weights],  # 对数据加权
                            input_length=int(max_tokens)))  # pading的长度

    model.add(Bidirectional(LSTM(units=int(num_1), return_sequences=True)))
    model.add(LSTM(units=int(num_2), return_sequences=False))
    model.add(Dense(units=1,  # 输出层1个神经元 1代表正面 0代表负面
                        activation='sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    train_history = model.fit(p_X_train, p_y_train, batch_size=batch_size, epochs=n_epoch,
                                  validation_data=(p_X_test, p_y_test))
    model.save('./models/emotion_model_LSTM.h5')

