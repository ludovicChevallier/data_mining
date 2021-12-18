import json
dataset={}
with open("./dataset/true_condition.json") as jsonFile:
    data=json.load(jsonFile)
    print(data)
    jsonFile.close()
dataset["Conditions"]=data
with open("./dataset/therapy.json") as jsonFile:
    data=json.load(jsonFile)
    print(data)
    jsonFile.close()
dataset["Therapies"]=data
with open("./dataset/patients.json") as jsonFile:
    data=json.load(jsonFile)
    print(data)
    jsonFile.close()
dataset["Patients"]=data

with open('./dataset/dataset.json', 'w') as fp:
    json.dump(dataset, fp)
