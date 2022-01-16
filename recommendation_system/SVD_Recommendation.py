import json
import pandas as pd
from surprise import KNNWithMeans,accuracy,KNNWithZScore,KNNBaseline,SlopeOne,SVD,CoClustering,SlopeOne,SVDpp
from surprise import Dataset
from surprise import Reader
import numpy as np 
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV,cross_validate
import random
#https://realpython.com/build-recommendation-engine-collaborative-filtering/
#https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-1-knn-item-based-collaborative-filtering-637969614ea

def Utility_matrix(data,id_patient):
    therapy=data["Therapies"]
    data=data["Patients"]
    therapy=pd.DataFrame(therapy,columns=["id","name","type"])
    patients_df=pd.DataFrame(data,columns=["age","birthdate","blood_group","conditions","country_of_residence","email","gender","id","name","occupation","trials","type"])
    test_cases=[]
    list_rating_train=[]
    list_item_train=[]
    list_user_train=[]
    for j in patients_df.index:
        if(patients_df["id"][j]==id_patient):
            test_cases.append(patients_df.iloc[j])
        for trial in patients_df["trials"][j]:
            kind=""
            for condition in patients_df["conditions"][j]:
                if(trial["condition"]==condition["id"]):
                    kind=condition["kind"]
            list_user_train.append(patients_df["id"][j])
            list_rating_train.append(int(trial["successful"]))                
            list_item_train.append(kind+"/"+trial["therapy"])       
    ratings_dict = {"user": list_user_train,"item": list_item_train,"rating": list_rating_train}
    print("DATA DONE")
    return ratings_dict,test_cases,therapy
def load_json():
    with open("./dataset/datasetB.json") as jsonFile:
        data=json.load(jsonFile)
        jsonFile.close()
    return data

def main(dataset,id_patient,id_condition):
    ratings_dict,test_cases,therapy=Utility_matrix(dataset,id_patient)
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(0, 100))
    data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
    trainingSet = data.build_full_trainset()
    algo = SVD()
    #n_factors =100, reg_all= 0.1, lr_all= 0.1,init_std_dev=0.2
    algo.fit(trainingSet)
    for test in test_cases:
        list_trials=[]
        kind=""
        pred_trials={}
        for condition in test["conditions"]:
            if(condition["id"]==id_condition):
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
                id_therapy=key.split("/")[1]
                t=id_therapy.split("Th")[1]
                t=int(t)-1
                name=therapy["name"][int(t)]
                print("the best therapy for the user "+str(test["id"])+":"+test["name"]+" is the therapy "+id_therapy+":"+name+" with a value of "+str(pred_trials[key]))
                j+=1
            else:
                break
        print("---------------")
    #trainset, testset = train_test_split(data, test_size=0.25)
    #predictions = algo.fit(trainset).test(testset)
    #print("GENERAL RMSE")
    #accuracy.rmse(predictions)
data=load_json()
id_patient=[6,51345,82486,51348,51358,51362,51366,51387,51416,51453]
id_condition=["pc32","pc277636","pc445475","pc277652","pc277696","pc277711","pc277723","pc277825","pc277986","pc278191"]
for i in range(len(id_patient)):
    main(data,id_patient[i],id_condition[i])

#def test_set(dataset):
    



