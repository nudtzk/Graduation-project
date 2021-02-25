
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import mean_absolute_error  #平均绝对误差，用于评估预测结果和真实数据集的接近程度的程度其其值越小说明拟合效果越好。
from sklearn.metrics import mean_squared_error  #均方差，该指标计算的是拟合数据和原始数据对应样本点的误差的平方和的均值，其值越小说明拟合效果越好。
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

all_data = pd.read_csv('D:\\Pythondata\\wcr.csv',header=0,index_col=0)   #读取文件 第一行当表头和第一列当索引
data_test=all_data[all_data.index <=  19]   #利用索引选取第一个文件做测试集
data_train=all_data[all_data.index > 19]  #选取后面所有文件做训练集
data_train= normalize(data_train)     #归一化
data_test= normalize(data_test)
data_train=pd.DataFrame(data_train)   #列表格式转化为dataframe格式
data_test=pd.DataFrame(data_test)
X_train=data_train.iloc[:, 0:10]   #训练集中选取前10列
y_train=data_train.iloc[:, [10]]   #训练集中选取第11列

clf_tree = tree.DecisionTreeRegressor()#实例化，建立评估模型对象，实例化用到的参数
clf_tree.fit(X_train, y_train)#模型接口，训练集   训练集数据训练模型

X_test=data_test.iloc[:, 0:10]    #测试集中选取前10列
y_test=data_test.iloc[:, [10]]    #测试集中选取第11列


y_test_pred = clf_tree.predict(X_test)  #测试集预测 
print(y_test_pred)
print(y_test)
print(mean_absolute_error(y_test, y_test_pred))
print(mean_squared_error(y_test, y_test_pred))

#画图
deflection=data_test.iloc[:,[9]]    #测试集第10列作为x轴
resistance=data_test.iloc[:, [10]]  #测试集第11列作为y轴
pred_resistance=y_test_pred   #训练结果作为y轴进行对比
plt.scatter(deflection,resistance,s=5,label="Original data")
plt.scatter(deflection,pred_resistance,s=5,label="Predicted data")
plt.ylim(0,1)
plt.title("Decision Tree Regression")
plt.xlabel("deflection")
plt.ylabel("resistance")
plt.legend()
plt.show()