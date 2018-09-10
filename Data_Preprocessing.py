# Setp 1 导入库
import numpy as np
import pandas as pd

# Step 2 导入数据
dataset = pd.read_csv("../100-Days-Of-ML-Code/datasets/Data.csv")
print(dataset)
X = dataset.iloc[:,:-1].values #iloc表示取数据集中的某些行和某些列，
#逗号前表示行，逗号后表示列，这里表示取所有行，列取除了最后一列的所有列，因为列是应变量
Y = dataset.iloc[:,3].values

# Step 3 整理数据
from sklearn.preprocessing import Imputer
#使用sklearn.preprocessing.Imputer类来处理使用np.nan对缺失值进行编码过的数据集。
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
#print(imputer)
#print(X[:,1:3])
imputer = imputer.fit(X[:,1:3])
#使用数组X去“训练”一个Imputer类，然后用该类的对象去处理数组Y中的缺失值，缺失值的处理方式是使用X中的均值（axis=0表示按列进行）代替Y中的缺失值。
#print(imputer)
X[:,1:3] = imputer.transform(X[:,1:3])
#print(X[:,1:3])

# Step 4 将标签解析成数字
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#LabelEncoder用于将字符转换成数字
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])#根据转换成的数字的种类使用OneHotEncoder重新编码
# 创建虚拟变量
#print(X)
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
#print("------------------------------")
#print("X")
#print(X)
#print("Y")
#print(Y)

# Step 5 拆分数据集 80:20
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)
#test_size：如果是浮点数，在0-1之间，表示样本占比；如果是整数的话就是样本的数量
#random_state：是随机数的种子。在需要重复试验的时候，保证得到一组一样的随机数。
# 比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
print("---------------------------------")
print("Step 5  Splitting the datasets into training sets and test sets")
print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("Y_train")
print(Y_train)
print("Y_test")
print(Y_test)

# Step 6 特征缩放
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
'''二者的功能都是对数据进行某种统一处理（比如标准化~N(0,1)，将数据缩放(映射)到某个固定区间，归一化，正则化等）
fit_transform(partData)对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），
然后对该partData进行转换transform，从而实现数据的标准化、归一化等等。。
根据对之前部分fit的整体指标，对剩余的数据（restData）使用同样的均值、方差、最大最小值等指标进行转换transform(restData)，
从而保证part、rest处理方式相同。
必须先用fit_transform(partData)，之后再transform(restData)
如果直接transform(partData)，程序会报错
如果fit_transfrom(partData)后，使用fit_transform(restData)而不用transform(restData)，
虽然也能归一化，但是两个结果不是在同一个“标准”下的，具有明显差异。
'''
print("--------------------------------")
print("Step 6 Feature Scaling")
print("X_train")
print(X_train)
print("X_test")
print(X_test)