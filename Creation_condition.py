import json
import pandas as pd

def read_condition():
    f = open('./dataset/condition.json', encoding="utf8")
    data = json.load(f)
    dataset=[]
    for i in range(len(data["sections"])):
        dataset.append(["C"+str(i),data["sections"][i]["name"]])
    dataset_JSON=pd.DataFrame(dataset,columns=["id","name"])
    dataset_JSON.to_json("./dataset/true_condition.json",orient="records")
    f.close()

read_condition()

