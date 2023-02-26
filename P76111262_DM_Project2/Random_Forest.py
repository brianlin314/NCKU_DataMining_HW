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
# 隨機森林(分類器)
# 隨機森林其實就是進階版的決策樹。隨機森林是使用 Bagging + 隨機特徵的技術所產生出來的 Ensemble learning 演算法。
# 
# Parameters:
# 。n_estimators: 森林中樹木的數量，預設=100。
# 。max_features: 劃分時考慮的最大特徵數，預設auto。
# 。criterion: 亂度的評估標準，gini/entropy。預設為gini。
# 。max_depth: 樹的最大深度。
# 。splitter: 特徵劃分點選擇標準，best/random。預設為best。
# 。random_state: 亂數種子，確保每次訓練結果都一樣，splitter=random 才有用。
# 。min_samples_split: 至少有多少資料才能再分
# 。min_samples_leaf: 分完至少有多少資料才能分
# 
# Attributes:
# 。feature_importances_: 查詢模型特徵的重要程度。
# 
# Methods:
# 。fit: 放入X、y進行模型擬合。
# 。predict: 預測並回傳預測類別。
# 。score: 預測成功的比例。
# 。predict_proba: 預測每個類別的機率值。
# 。get_depth: 取得樹的深度。

# %%
from sklearn.ensemble import RandomForestClassifier

# 建立 Random Forest Classifier 模型
randomForestModel = RandomForestClassifier(n_estimators=100, criterion = 'gini')
# 使用訓練資料訓練模型
randomForestModel.fit(X_train, y_train)
# 預測成功的比例
print('訓練集準確率: ',randomForestModel.score(X_train,y_train))
print('測試集準確率: ',randomForestModel.score(X_test,y_test))

# %%
print('特徵重要程度: ',randomForestModel.feature_importances_)

# %%
from sklearn import tree
feature=['age','gender','height','weight','attitude','prepare_hour','class_type','pass_rate','glasses','classhw','care']
className=['good','bad']
fig, axes = plt.subplots(nrows = 1,ncols = 4,figsize = (10,2), dpi=900)
# 繪製前五棵樹
for index in range(0, 4):
    tree.plot_tree(randomForestModel.estimators_[index],
                   feature_names = feature, 
                   class_names=className,
                   filled = True,
                   ax = axes[index]);

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
y_prediction = randomForestModel.predict(X_test)
mat = confusion_matrix(y_test, y_prediction)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=[0, 1], yticklabels=[0,1])
plt.xlabel('true label')
plt.ylabel('predicted label');

plt.show()

# %%
df_alter=pd.read_csv("./teacher_data_alter.csv") #讀csv

label_encoder=LabelEncoder()
df_alter["gender"]=label_encoder.fit_transform(df_alter["gender"]) # 對gender做label Encoder
df_alter["attitude"]=label_encoder.fit_transform(df_alter["attitude"]) # 對attitude做label Encoder
df_alter["class_type"]=label_encoder.fit_transform(df_alter["class_type"]) # 對class_type做label Encoder
df_alter["pass_rate"]=label_encoder.fit_transform(df_alter["pass_rate"]) # 對pass_rate做label Encoder
df_alter["classhw"]=label_encoder.fit_transform(df_alter["classhw"]) # 對classhw做label Encoder
df_alter["care"]=label_encoder.fit_transform(df_alter["care"]) # 對care做label Encoder
df_alter["label"]=label_encoder.fit_transform(df_alter["label"]) # 對label做label Encoder

X_alter=df_alter.drop(labels=["label"],axis=1).values
y_alter=df_alter["label"].values
print("微調後資料集準確率:",randomForestModel.score(X_alter,y_alter)) #使用測試資料預測分類，並印出準確率

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
y_prediction_alter = randomForestModel.predict(X_alter)
mat = confusion_matrix(y_alter, y_prediction_alter)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=[0, 1], yticklabels=[0,1])
plt.xlabel('True label')
plt.ylabel('Predicted label')

plt.show()

# %%



