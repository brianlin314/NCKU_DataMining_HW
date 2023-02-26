# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests

# %%
df=pd.read_csv("./teacher_data.csv") #讀csv
print(df)

# %%
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
df["gender"]=label_encoder.fit_transform(df["gender"]) # 對gender做label Encoder
df["attitude"]=label_encoder.fit_transform(df["attitude"]) # 對attitude做label Encoder
df["class_type"]=label_encoder.fit_transform(df["class_type"]) # 對class_type做label Encoder
df["pass_rate"]=label_encoder.fit_transform(df["pass_rate"]) # 對pass_rate做label Encoder
df["classhw"]=label_encoder.fit_transform(df["classhw"]) # 對classhw做label Encoder
df["care"]=label_encoder.fit_transform(df["care"]) # 對care做label Encoder
df["label"]=label_encoder.fit_transform(df["label"]) # 對label做label Encoder
print(df)

# %%
df=df.dropna()
print("當前資料缺失值總數:",len(np.where(np.isnan(df))[0]))

# %%
from sklearn.model_selection import train_test_split
X=df.drop(labels=["label"],axis=1).values #沒label
y=df["label"].values #有label
print("X:\n",X)
print("y:\n",y)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
print("Training data shape:",X_train.shape)
print("Testing data shape:",X_test.shape)

# %% [markdown]
# 建立 k-nearest neighbors(KNN) 模型
# 
# Parameters:
# 。n_neighbors: 設定鄰居的數量(k)，選取最近的k個點，預設為5。
# 。algorithm: 搜尋數演算法{'auto'，'ball_tree'，'kd_tree'，'brute'}，可選。
# 。metric: 計算距離的方式，預設為歐幾里得距離。
# 
# Attributes:
# 。classes_: 取得類別陣列。
# 。effective_metric_: 取得計算距離的公式。
# 
# Methods:
# 。fit: 放入X、y進行模型擬合。
# 。predict: 預測並回傳預測類別。
# 。score: 預測成功的比例。

# %%
from sklearn.manifold import TSNE # 進行TSNE降維，降成2維

tsneModel = TSNE(n_components=2, random_state=42,n_iter=1000)
train_reduced = tsneModel.fit_transform(X_train)
test_reduced = tsneModel.fit_transform(X_test)

# %%
plt.figure(figsize=(8,6))
plt.scatter(train_reduced[:, 0], train_reduced[:, 1], c=y_train, alpha=0.5,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))

plt.colorbar()
plt.show()

# %%
from sklearn.neighbors import KNeighborsClassifier

# 建立KNN模型
knnModel = KNeighborsClassifier(n_neighbors=2)
# 使用訓練資料訓練模型
knnModel.fit(train_reduced,y_train) # 降維資料
# 預測成功的比例
print('Training Set Accuracy: ',knnModel.score(train_reduced,y_train))
print('Testing Set Accuracy: ',knnModel.score(test_reduced,y_test))


