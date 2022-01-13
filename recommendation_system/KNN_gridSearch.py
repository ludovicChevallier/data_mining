import json
import pandas as pd
from surprise import KNNWithMeans,accuracy,KNNWithZScore,KNNBaseline,SlopeOne,SVD
from surprise import Dataset
from surprise import Reader
from sklearn.neighbors import NearestNeighbors
import numpy as np 
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
#https://realpython.com/build-recommendation-engine-collaborative-filtering/
#https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-1-knn-item-based-collaborative-filtering-637969614ea

def load_json():
    with open("./dataset/datasetB.json") as jsonFile:
        data=json.load(jsonFile)
        jsonFile.close()
    data=data["Patients"]
    patients_df=pd.DataFrame(data,columns=["age","birthdate","blood_group","conditions","country_of_residence","email","gender","id","name","occupation","trials","type"])
    list_rating=[]
    list_item=[]
    list_user=[]
    test_cases=[]
    list_rating_train=[]
    list_item_train=[]
    list_user_train=[]
    id_patient=[6,51345,82486,51348,51358,51362,51366,51387,51416,51453]
    id_condition=["pc32","pc277636","pc445475","pc277652","pc277696","pc277711","pc277723","pc277825","pc277986","pc278191"]
    condition_test=""
    
    for j in patients_df.index:
        if(j<=5000):
            if(patients_df["id"][j] in id_patient ):
                #test_cases.append(patients_df.iloc[j])
                condition_test=id_condition[id_patient.index(patients_df["id"][j])]
            for trial in patients_df["trials"][j]:
                kind=""
                for condition in patients_df["conditions"][j]:
                    if(trial["condition"]==condition["id"]):
                        kind=condition["kind"]
                #if(patients_df["id"][j] in id_patient and trial["condition"]!=condition_test):
                    #getting only the therapy for the condition that are not link to the condition we try to heal.
                    #This will be use to test our test cases
                    #list_user.append(patients_df["id"][j])
                    #list_rating.append(int(trial["successful"]))                
                    #list_item.append(kind+"/"+trial["therapy"])
                #else:
                if(kind==""):
                    print("ERROR")
                list_user_train.append(patients_df["id"][j])
                list_rating_train.append(int(trial["successful"]))                
                list_item_train.append(kind+"/"+trial["therapy"])        
    test_dict={"user": list_user,"item": list_item,"rating": list_rating}
    ratings_dict = {"user": list_user_train,"item": list_item_train,"rating": list_rating_train}
    return ratings_dict,test_cases,test_dict

def main():
    ratings_dict,test_cases,test_dict=load_json()
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(0, 100))
    data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
    sim_options = {
        'k':[1,2,10,30,50],
        'name': [ 'cosine','pearson'],
        'min_support': [10,20,30],
        'user_based': [False],
        }
    bsl_options = {'method': ['als', 'sgd'],
               'n_epochs': [20,30,40],
               }
              
    #,"bsl_options":bsl_options
    param_grid = {"sim_options": sim_options}
    #parameters of gridSeachCV:https://surprise.readthedocs.io/en/stable/model_selection.html
    gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse", "mae"], cv=5)
    gs.fit(data)
    print(gs.best_score["rmse"])
    print(gs.best_params["rmse"])
main()