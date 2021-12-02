import csv
import pandas as pd

def read_therapy():
    f = open('List_therapy.csv')
    data = csv.reader(f)
    dataset=[]
    i=0
    for item in data:
        dataset.append(["T"+str(i),item])
        i+=1
    dataset_JSON=pd.DataFrame(dataset,columns=["id","name"])
    dataset_JSON.to_json("./dataset/therapy.json",orient="records")
    f.close()

read_therapy()