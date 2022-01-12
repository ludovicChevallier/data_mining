import json
import pandas as pd
from surprise import KNNWithMeans,accuracy,KNNWithZScore,KNNBaseline,SlopeOne,SVD,CoClustering,SlopeOne,SVDpp
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
                list_user_train.append(patients_df["id"][j])
                list_rating_train.append(int(trial["successful"]))                
                list_item_train.append(kind+"/"+trial["therapy"])       
    test_dict={"user": list_user,"item": list_item,"rating": list_rating}
    ratings_dict = {"user": list_user_train,"item": list_item_train,"rating": list_rating_train}
    print("DATA DONE")
    return ratings_dict,test_cases,test_dict

def main():
    ratings_dict,test_cases,test_dict=load_json()
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(0, 100))
    data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
    sim_options = {
    'n_factors': 500,
    'reg_all': 0.1,
    'lr_all': 0.001,
    'init_std_dev': 0.2
    }   
    param_grid =sim_options
    #parameters of gridSeachCV:https://surprise.readthedocs.io/en/stable/model_selection.html
    algo = SVD(n_factors= 250,reg_all=0.1)
    trainset, testset = train_test_split(data, test_size=0.25)
    predictions = algo.fit(trainset).test(testset)
    print("GENERAL RMSE")
    accuracy.rmse(predictions)

main()