import json
import pandas as pd
from surprise import KNNWithMeans,accuracy,KNNWithZScore,KNNBaseline,SlopeOne,SVD,CoClustering,SlopeOne,SVDpp
from surprise import Dataset
from surprise import Reader
from sklearn.neighbors import NearestNeighbors
import numpy as np 
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV,cross_validate
import collections
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
    
    for j in patients_df.index:
        if(patients_df["id"][j] in id_patient ):
            test_cases.append(patients_df.iloc[j])
        for trial in patients_df["trials"][j]:
            kind=""
            for condition in patients_df["conditions"][j]:
                if(trial["condition"]==condition["id"]):
                    kind=condition["kind"]
            if(kind==""):
                print("ERROR")
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
    trainingSet = data.build_full_trainset()
    sim_options = {
    'n_factors': 500,
    'reg_all': 0.1,
    'lr_all': 0.001,
    'init_std_dev': 0.2
    }   
    param_grid =sim_options
    #parameters of gridSeachCV:https://surprise.readthedocs.io/en/stable/model_selection.html
    algo = SVD()
    #n_factors =100, reg_all= 0.1, lr_all= 0.1,init_std_dev=0.2
    algo.fit(trainingSet)
    x=0
    id_patient=[6,51345,82486,51348,51358,51362,51366,51387,51416,51453]
    id_condition=["pc32","pc277636","pc445475","pc277652","pc277696","pc277711","pc277723","pc277825","pc277986","pc278191"]
    for test in test_cases:
        list_trials=[]
        kind=""
        pred_trials={}
        x=id_patient.index(test["id"])
        for condition in test["conditions"]:
            if(condition["id"]==id_condition[x]):
                kind=condition["kind"]
        for i in range(len(test["trials"])):
            list_trials.append(test["trials"][i]["therapy"])
        for elem in df["item"]:
            if(elem.split("/")[1] not in list_trials and elem.split("/")[0]==kind):
                prediction = algo.predict(test["id"],elem)
                val=np.round(np.clip(prediction.est,0,100))
                pred_trials[elem]=val
        keylist = sorted(pred_trials, key=pred_trials.get,reverse=True)
        j=0
        for key in keylist:
            if(j<=4):
                print("the best therapy for the user "+test["name"]+" is the therapy: "+key+" with a value of "+str(pred_trials[key]))
                j+=1
            else:
                break
        print("---------------")
    trainset, testset = train_test_split(data, test_size=0.25)
    predictions = algo.fit(trainset).test(testset)
    print("GENERAL RMSE")
    accuracy.rmse(predictions)
    #cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)

main()