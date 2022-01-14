import json
import pandas as pd
from surprise import KNNWithMeans,accuracy,KNNWithZScore,KNNBaseline,SlopeOne,SVD
from surprise import Dataset
from surprise import Reader
from sklearn.neighbors import NearestNeighbors
import numpy as np 
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
def load_json():
    with open("./dataset/datasetB.json") as jsonFile:
        data=json.load(jsonFile)
        jsonFile.close()
    data=data["Patients"]
    patients_df=pd.DataFrame(data,columns=["age","birthdate","blood_group","conditions","country_of_residence","email","gender","id","name","occupation","trials","type"])

    list_rating_train=[]
    list_item_train=[]
    list_user_train=[]
    id_patient=[6,51345,82486,51348,51358,51362,51366,51387,51416,51453]
    id_condition=["pc32","pc277636","pc445475","pc277652","pc277696","pc277711","pc277723","pc277825","pc277986","pc278191"]
    condition_test=""
    non_condition=0
    for j in patients_df.index:
            if(patients_df["id"][j] in id_patient ):
                condition_test=id_condition[id_patient.index(patients_df["id"][j])]
            for trial in patients_df["trials"][j]:
                kind=""
                for condition in patients_df["conditions"][j]:
                    if(trial["condition"]==condition["id"]):
                        kind=condition["kind"]
                list_user_train.append(patients_df["id"][j])
                list_rating_train.append(str(patients_df["id"][j])+"/"+kind+"/"+trial["therapy"])                
                list_item_train.append(kind+"/"+trial["therapy"])
            if(len(patients_df["trials"][j])==0):
                non_condition+=1

    ratings_dict = {"user": list_user_train,"item": list_item_train,"rating": list_rating_train}
    nb_therapy=len(set(list_item_train))
    nb_user=len(set(list_user_train))
    actual_combinaison=len(set(list_rating_train))
    total_comb=nb_therapy*nb_user
    print((total_comb-actual_combinaison)/total_comb)
    #we have 79% of emptinesse when we have as a utility matrix : user and therapy
    #we have 99% of emptinesse when we have as a utility matrix : user and cond/therapy
    numb_duplicates=(len(list_rating_train)-len(set(list_rating_train)))/len(list_rating_train)
    #2% of duplicates
    print(numb_duplicates)
def load_json_own_DB():
    with open("./dataset/dataset.json") as jsonFile:
        data=json.load(jsonFile)
        jsonFile.close()
    data=data["Patients"]
    patients_df=pd.DataFrame(data,columns=["id","name","conditions","trials"])
    list_rating_train=[]
    list_item_train=[]
    list_user_train=[]
    test_cases=[]
    for j in patients_df.index:       
        for trial in patients_df["trials"][j]:
            kind=""
            for condition in patients_df["conditions"][j]:
                if(condition["id"]==trial["condition"]):
                    kind=condition["kind"]
            list_user_train.append(patients_df["id"][j])
            list_rating_train.append(str(patients_df["id"][j])+"/"+kind+"/"+trial["therapy"])
            list_item_train.append(kind+"/"+trial["therapy"])
    ratings_dict = {"user": list_user_train,"item": list_item_train,"rating": list_rating_train}
    nb_therapy=len(set(list_item_train))
    nb_user=len(set(list_user_train))
    actual_combinaison=len(set(list_rating_train))
    total_comb=nb_therapy*nb_user
    print((total_comb-actual_combinaison)/total_comb)
    #we have 93% of emptinesse when we have as a utility matrix : user and therapy
    #we have 99% of emptinesse when we have as a utility matrix : user and cond/therapy
    numb_duplicates=(len(list_rating_train)-len(set(list_rating_train)))/len(list_rating_train)
    #0% of duplicates
    print(numb_duplicates)
load_json_own_DB()
