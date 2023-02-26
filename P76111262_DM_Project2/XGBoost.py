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

# %%
from xgboost import XGBClassifier

# 建立 XGBClassifier 模型
xgboostModel = XGBClassifier(n_estimators=100, learning_rate= 0.3)
# 使用訓練資料訓練模型
xgboostModel.fit(X_train, y_train)
# 預測成功的比例
print('訓練集準確率: ',xgboostModel.score(X_train,y_train))
print('測試集準確率: ',xgboostModel.score(X_test,y_test))

# %%
from xgboost import plot_importance
from xgboost import plot_tree

plot_importance(xgboostModel)
print('特徵重要程度: ',xgboostModel.feature_importances_)

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
y_prediction = xgboostModel.predict(X_test)
mat = confusion_matrix(y_test, y_prediction)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=[0, 1], yticklabels=[0,1])
plt.xlabel('true label')
plt.ylabel('predicted label')

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
print("微調後資料集準確率:",xgboostModel.score(X_alter,y_alter)) #使用測試資料預測分類，並印出準確率

# %%
from sklearn.metrics import confusion_matrix
import seaborn as sns
y_prediction_alter = xgboostModel.predict(X_alter)
mat = confusion_matrix(y_alter, y_prediction_alter)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=[0, 1], yticklabels=[0,1])
plt.xlabel('True label')
plt.ylabel('Predicted label')

plt.show()


