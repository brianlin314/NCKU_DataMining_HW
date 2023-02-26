import pandas as pd
import random
import sklearn as sk
features=["age","gender","height","weight","attitude","prepare_hours","class_type","pass_rate","glasses","classhw","care","label"]
data=pd.DataFrame(columns=features)
#print(data)
positive=0
negative=0
for i in range(3000):
    temp=[random.randint(25,65), random.choice(["M","F"]),random.randint(140,190),random.randint(40,100),
        random.choice(["Serious","Free"]),random.randint(0,6),random.choice(["Online","Physical"]),
        random.choice(["Low","High"]),random.randint(0,1),random.choice(["Low","Middle","High"]),random.choice(["Yes","No"])]
    if temp[4]=="Serious" and (temp[5]==3 or temp[5]==4 or temp[5]==5 or temp[5]==6) and temp[6]=="Physical" and temp[7]=="High" and temp[10]=="Yes":
        temp.append("good")
        positive+=1
    else:
        temp.append("bad")
        negative+=1
    data.loc[i]=temp

for i in range(3000,6000):
    temp=[random.randint(25,65), random.choice(["M","F"]),random.randint(140,190),random.randint(40,100),
        "Serious",random.randint(3,6),"Physical",
        "High",random.randint(0,1),random.choice(["Low","Middle","High"]),"Yes","good"]
    positive+=1
    data.loc[i]=temp

data_shuffled=sk.utils.shuffle(data)
print(data_shuffled)
print("positive",positive,"/","negative",negative)
data_shuffled.to_csv("teacher_data.csv",index=False)