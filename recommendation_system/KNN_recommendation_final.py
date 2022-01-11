import json
import pandas as pd
from surprise import KNNWithMeans,accuracy
from surprise import Dataset
from surprise import Reader
from sklearn.neighbors import NearestNeighbors
import numpy as np 
from surprise.model_selection import train_test_split
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
    condition=""
    
    for j in patients_df.index:
        if(patients_df["id"][j] in id_patient ):
            #test_cases.append(patients_df.iloc[j])
            condition=id_condition[id_patient.index(patients_df["id"][j])]
        if(j==0):
            print(patients_df["trials"][j])
        for trial in patients_df["trials"][j]:
                if(patients_df["id"][j] in id_patient and trial["condition"]!=condition):
                    #getting only the therapy for the condition that are not link to the condition we try to heal.
                    #This will be use to test our test cases
                    list_user.append(patients_df["id"][j])
                    list_rating.append(int(trial["successful"]))                
                    list_item.append(trial["therapy"])
                else:
                    list_user_train.append(patients_df["id"][j])
                    list_rating_train.append(int(trial["successful"]))                
                    list_item_train.append(trial["therapy"])


            
    test_dict={"user": list_user,"item": list_item,"rating": list_rating}
    ratings_dict = {"user": list_user_train,"item": list_item_train,"rating": list_rating_train}
    return ratings_dict,test_cases,test_dict

def main():
    ratings_dict,test_cases,test_dict=load_json()
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(0, 100))
    data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
    sim_options = {
        "name": "cosine", #use cosine similarity
        "min_support": 3,
        "user_based": False,  # Compute  similarities between user
    }
    trainingSet = data.build_full_trainset()
    algo = KNNWithMeans(sim_options=sim_options)
    algo.fit(trainingSet)
    for test in test_cases:
        max_value=0.0
        best_thera=""
        list_trials=[]
        for i in range(len(test["trials"])):
            list_trials.append(test["trials"][i]["therapy"])
        for elem in df["item"]:
            if(elem not in list_trials):
                prediction = algo.predict(test["id"],elem)
                val=np.round(np.clip(prediction.est,0,100))
                if(val>max_value):
                    max_value=val
                    best_thera=elem
        print("the best therapy for the user "+test["name"]+" is the therapy: "+best_thera+" with a value of "+str(max_value))
    #General RMSE
    trainset, testset = train_test_split(data, test_size=0.25)
    predictions = algo.fit(trainset).test(testset)
    print("GENERAL RMSE")
    accuracy.rmse(predictions)
    #test of rmse for the test_cases
    df = pd.DataFrame(test_dict)
    reader = Reader(rating_scale=(0, 100))
    data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
    testset = [data.df.loc[i].to_list() for i in range(len(data.df))]
    prediction=algo.test(testset)
    print("RMSE FOR THE TESTCASES")
    accuracy.rmse(prediction)
main()