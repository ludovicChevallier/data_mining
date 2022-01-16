import json
from tkinter.filedialog import SaveFileDialog
import pandas as pd
from surprise import KNNWithMeans,accuracy,KNNWithZScore,KNNBaseline,SlopeOne,SVD,CoClustering,SlopeOne,SVDpp,NMF
from surprise import Dataset
from surprise import Reader
import numpy as np 
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV,cross_validate
import random


def load_json():
    with open("./dataset/datasetB.json") as jsonFile:
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
    i=0
    list_id={}
    for j in patients_df.index:
        list_condition = sorted(patients_df["conditions"][j], key=lambda d: d['diagnosed'],reverse=True)
        
        if(len(list_condition)!=0 and list_condition[0]["cured"]!="Null" and list_condition[0]["isTreated"]!=False):
            list_id[j]=list_condition[0]['diagnosed']
    list_id = sorted(list_id.items(), key=lambda d: d[1] ,reverse=True)
    list_id=[ x for (x,y) in list_id[:5000]]    
    #contains the 100000 id with most recent cured conditions
    for j in patients_df.index:
            #(1)delete the youngest condition if it's not cured
            list_condition = sorted(patients_df["conditions"][j], key=lambda d: d['diagnosed'],reverse=True) 
            #It's possible that some patient doesn't have any condition
            if(len(list_condition)!=0):
                while(len(list_condition)!=0 and list_condition[0]["cured"]=="Null"):
                    del list_condition[0]
                try:
                    last_condition=list_condition[0]
                except:
                    continue
            else:
                #If there is no condition inside this patient we continue
                continue
            #Now we hide the therapy that cured the youngest condition
            for trial in patients_df["trials"][j]:
                kind=""
                for condition in list_condition:
                    if(trial["condition"]==condition["id"]):
                        kind=condition["kind"]
                if(kind!=""):
                    if(j in list_id):
                        #It means that we take that example in the test_set
                        if(trial["condition"]==last_condition["id"] and trial["successful"]==100):
                            test_cases.append(patients_df.iloc[j])
                            list_user_test.append(patients_df["id"][j])
                            list_rating_test.append(int(trial["successful"]))                
                            list_item_test.append(kind+"/"+trial["therapy"])  
                        else:
                            list_user_train.append(patients_df["id"][j])
                            list_rating_train.append(int(trial["successful"]))                
                            list_item_train.append(kind+"/"+trial["therapy"]) 
                    else:
                        list_user_train.append(patients_df["id"][j])
                        list_rating_train.append(int(trial["successful"]))                
                        list_item_train.append(kind+"/"+trial["therapy"])
    ratings_dict = {"user": list_user_train,"item": list_item_train,"rating": list_rating_train}
    test_dict={"user": list_user_test,"item": list_item_test,"rating": list_rating_test}
    return ratings_dict,test_cases,therapy,test_dict

def main(dataset,k):
    ratings_dict,test_cases,therapy,test_dict=Building_Utility_Matrix_test(dataset)
    print("DATA DONE")
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(0, 100))
    data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
    trainingSet = data.build_full_trainset()
    algo = SVD(n_factors =1000)
    #n_factors =100, reg_all= 0.1, lr_all= 0.1,init_std_dev=0.2
    print("START FIT")
    algo.fit(trainingSet)
    print("FIT DONE")
    successful_patients=[]
    i=0
    for test in test_cases:
        #kind=""
        pred_trials={}
        x=test_dict["user"].index(test["id"])
        #for condition in test["conditions"]:
            #We want to predict for the last condition
            #if(condition["kind"]==test_dict["item"][x].split("/")[0]):
        kind=test_dict["item"][x].split("/")[0]
        for thera in therapy:
            elem=kind+"/"+thera["id"]
            prediction = algo.predict(test["id"],elem)
            val=np.round(np.clip(prediction.est,0,100))
            pred_trials[thera["id"]]=val
        if(i%100==0):
            print(sorted(pred_trials.items(), key=lambda d: d[1] ,reverse=True))
        keylist = [x for (x,y) in sorted(pred_trials.items(), key=lambda d: d[1] ,reverse=True)]
        if test_dict["item"][x].split("/")[1] in keylist[:k] : 
            successful_patients.append(test_dict["user"])
        i+=1

    print(len(successful_patients)/len(test_cases))
dataset=load_json()
main(dataset,5)
#main(dataset,5)