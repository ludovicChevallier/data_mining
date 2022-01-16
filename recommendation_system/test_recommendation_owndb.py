import json
import pandas as pd
from surprise import KNNWithMeans,accuracy,KNNWithZScore,KNNBaseline,SlopeOne,SVD,CoClustering,SlopeOne,SVDpp
from surprise import Dataset
from surprise import Reader
import numpy as np 
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV,cross_validate
import random

def load_json():
    with open("./dataset/dataset.json") as jsonFile:
        data=json.load(jsonFile)
        jsonFile.close()
    return data

def Building_Utility_Matrix_test(data):
    therapy=data["Therapies"]
    list_rating_train=[]
    list_item_train=[]
    list_user_train=[]
    list_rating_test=[]
    list_item_test=[]
    list_user_test=[]
    test_cases=[]
    data=data["Patients"]
    patients_df=pd.DataFrame(data,columns=["age","birthdate","blood_group","conditions","country_of_residence","email","gender","id","name","occupation","trials","type"])
    #take randomly 20% of ids
    list_id=[]
    while(len(list_id)<200):
        id=random.randint(0,1000)
        while(id in list_id):
            id=random.randint(0,1000)
        list_id.append(id)
    i=0
    for j in patients_df.index:
            #(1)delete the youngest condition if it's not cured
            list_condition=patients_df["conditions"][j]
            for condition in list_condition:
                condition['diagnosed']=condition['diagnosed'].replace("-","")
                condition['diagnosed']=int(condition['diagnosed'])
            list_condition = sorted(list_condition, key=lambda d: d['diagnosed'],reverse=True) 
            #It's possible that some patient doesn't have any condition
            if(len(list_condition)!=0):
                while(len(list_condition)!=0 and list_condition[0]["cured"]==None):
                    del list_condition[0]
                if(len(list_condition)!=0):
                    last_condition=list_condition[0]
                else:
                    #If there is no condition inside this patient need to take an other patiet
                    if(j in list_id):
                        id=random.randint(j+1,1000)
                        while(id in list_id):
                            id=random.randint(j+1,1000)
                        list_id.append(id)
                        list_id.remove(j)
            else:
                #If there is no condition inside this patient need to take an other patiet
                if(j in list_id):
                        id=random.randint(j+1,1000)
                        while(id in list_id):
                            id=random.randint(j+1,1000)
                        list_id.append(id)
                        list_id.remove(j)
            #Now we hide the therapy that cured the youngest condition
            a=False
            for trial in patients_df["trials"][j]:
                kind=""
                for condition in list_condition:
                    if(trial["condition"]==condition["id"]):
                        kind=condition["kind"]
                #It means that we take that example in the test_set
                if(kind!=""):
                    if(j in list_id):
                        if(trial["condition"]==last_condition["id"]):
                            if(a==True):
                                if(int(trial["sucessful"].split("%")[0])> list_rating_test[-1]):
                                    list_rating_test[-1]=int(trial["sucessful"].split("%")[0])                
                                    list_item_test[-1]=kind+"/"+trial["therapy"]
                            else:
                                a=True
                                test_cases.append(patients_df.iloc[j])
                                list_user_test.append(patients_df["id"][j])
                                list_rating_test.append(int(trial["sucessful"].split("%")[0]))                
                                list_item_test.append(kind+"/"+trial["therapy"])  
                        else:
                            list_user_train.append(patients_df["id"][j])
                            list_rating_train.append(int(trial["sucessful"].split("%")[0]))                
                            list_item_train.append(kind+"/"+trial["therapy"]) 
                    else:
                        list_user_train.append(patients_df["id"][j])
                        list_rating_train.append(int(trial["sucessful"].split("%")[0]))                
                        list_item_train.append(kind+"/"+trial["therapy"])
            if(a==False and j in list_id):
                id=random.randint(j+1,100000)
                while(id in list_id):
                    id=random.randint(j+1,100000)
                list_id.append(id)
                list_id.remove(j)
    ratings_dict = {"user": list_user_train,"item": list_item_train,"rating": list_rating_train}
    test_dict={"user": list_user_test,"item": list_item_test,"rating": list_rating_test}
    return ratings_dict,test_cases,therapy,test_dict

def main(dataset,k):
    ratings_dict,test_cases,therapy,test_dict=Building_Utility_Matrix_test(dataset)
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(0, 100))
    data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
    trainingSet = data.build_full_trainset()
    algo = SVD()
    #n_factors =100, reg_all= 0.1, lr_all= 0.1,init_std_dev=0.2
    algo.fit(trainingSet)
    i=0
    count=0
    for test in test_cases:
        list_trials=[]
        kind=""
        pred_trials={}
        x=test_dict["user"].index(test["id"])
        for condition in test["conditions"]:
            #We want to predict for the last condition
            if(condition["kind"]==test_dict["item"][x].split("/")[0]):
                kind=condition["kind"]
        for i in range(len(test["trials"])):
            #It should not in consideration the previous therapy except the one that reached 100%
            if(test["trials"][i]["therapy"]!=test_dict["item"][x].split("/")[1]):
                list_trials.append(test["trials"][i]["therapy"])
        for elem in df["item"]:
            if(elem.split("/")[1] not in list_trials and elem.split("/")[0]==kind):
                prediction = algo.predict(test["id"],elem)
                val=np.round(np.clip(prediction.est,0,100))
                pred_trials[elem]=val
        keylist = sorted(pred_trials, key=pred_trials.get,reverse=True)
        j=1
        for key in keylist:
            if(j<=k):
                id_therapy=key.split("/")[1]
                if(id_therapy==test_dict["item"][x].split("/")[1]):
                    count+=1
                j+=1
            else:
                break
    print(count/len(test_cases))
dataset=load_json()
main(dataset,5)
#main(dataset,5)