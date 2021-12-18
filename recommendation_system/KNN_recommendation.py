import json
import pandas as pd
from surprise import KNNWithMeans
from surprise import Dataset
from surprise import Reader
from sklearn.neighbors import NearestNeighbors
import numpy as np 
#https://realpython.com/build-recommendation-engine-collaborative-filtering/
#https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-1-knn-item-based-collaborative-filtering-637969614ea

def load_json():
    with open("./dataset/dataset.json") as jsonFile:
        data=json.load(jsonFile)
        jsonFile.close()
    data=data["Patients"]
    patients_df=pd.DataFrame(data,columns=["id","name","conditions","trials"])
    list_rating=[]
    list_item=[]
    list_user=[]
    test_cases=[]
    for j in patients_df.index:
        if(patients_df["conditions"][j][0]["cured"]==None):
            test_cases.append(patients_df.iloc[j])
    for j in patients_df.index:       
        for trial in patients_df["trials"][j]:
            list_user.append(patients_df["id"][j])
            list_rating.append(int(trial["sucessful"].split("%")[0]))
            list_item.append(trial["therapy"])
    ratings_dict = {"user": list_user,"item": list_item,"rating": list_rating,}
    return ratings_dict,test_cases

def main():
    ratings_dict,test_cases=load_json()
    df = pd.DataFrame(ratings_dict)
    reader = Reader(rating_scale=(0, 100))
    data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)
    sim_options = {
        "name": "msd", #use cosine similarity
        "min_support": 3,
        "user_based": False,  # Compute  similarities between items
    }
    trainingSet = data.build_full_trainset()
    algo = KNNWithMeans(sim_options=sim_options)
    algo.fit(trainingSet)
    max_value=0.0
    best_thera=""

    for test in test_cases:
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

main()