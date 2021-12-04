import csv
import pandas as pd

def read_therapy():
    f = open('List_therapy.csv')
    data = csv.reader(f)
    dataset=[]
    i=0
    for item in data:
        types=item[0].split("therapy")[0]
        dataset.append(["T"+str(i),item[0],types])
        i+=1
    dataset_JSON=pd.DataFrame(dataset,columns=["id","name","type"])
    dataset_JSON.to_json("./dataset/therapy.json",orient="records")
    f.close()

read_therapy()