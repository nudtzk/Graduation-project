

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error  #平均绝对误差，用于评估预测结果和真实数据集的接近程度的程度其其值越小说明拟合效果越好。
from sklearn.metrics import mean_squared_error  #均方差，该指标计算的是拟合数据和原始数据对应样本点的误差的平方和的均值，其值越小说明拟合效果越好。
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
import shap



from sklearn.model_selection import KFold




all_data = pd.read_csv('/Users/yukinan/PycharmProjects/KunZhang/shearwall_data/all_data.csv',header=0,index_col=0)   #读取文件 第一行当表头和第一列当索引
column_name = list(all_data.columns)

# data_test=all_data[all_data.index <=  19]   #利用索引选取第一个文件做测试集
# data_train=all_data[all_data.index > 19]  #选取后面所有文件做训练集
# data_train= normalize(data_train)     #归一化
# data_test= normalize(data_test)
# data_train=pd.DataFrame(data_train)   #列表格式转化为dataframe格式
# data_test=pd.DataFrame(data_test)
all_data=pd.DataFrame(all_data)
all_data=pd.DataFrame(normalize(all_data))
X=all_data.iloc[:, 0:10]   #训练集中选取前10列
y=all_data.iloc[:, 10]   #训练集中选取第11列
scores = []
clf_tree = tree.DecisionTreeRegressor()
cv = KFold(n_splits=10, random_state=42, shuffle=False)
for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)
    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
    clf_tree.fit(X_train, y_train)
    scores.append(clf_tree.score(X_test, y_test))
    print("mean socre:", np.mean(scores))


# X_test=data_test.iloc[:, 0:10]    #测试集中选取前10列
# y_test=data_test.iloc[:, [10]]    #测试集中选取第11列


    y_test_pred = clf_tree.predict(X_test)  #测试集预测
    print(y_test_pred)
    print(y_test)
    print(mean_absolute_error(y_test, y_test_pred))
    print(mean_squared_error(y_test, y_test_pred))

# 画图
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

explainer = shap.TreeExplainer(clf_tree)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=column_name, plot_type='violin')
shap.summary_plot(shap_values, X_test, feature_names=column_name, plot_type='bar')
shap.dependence_plot("deflection", shap_values, X_test)
plt.show()
