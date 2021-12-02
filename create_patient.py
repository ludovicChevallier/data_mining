import names
import random
import json
import datetime
import pandas as pd

def create_patients():
    dataset=[]
    f = open('./dataset/true_condition.json', encoding="utf8")
    data = json.load(f)

    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 1)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days

    for j in range(100):
        nb_condition=random.randint(1,3)
        conditions=[]
        list_id_condition=[]
        #create a random list of condtions between 1 and 3
        for i in range(nb_condition):
            a=False
            #check that the conditions was not already taken
            while(a==False):
                id_condition=data[random.randint(0,len(data)-1)]["id"]
                if(id_condition not in list_id_condition):
                    list_id_condition.append(id_condition)
                    a=True
                
            #generate random date
            random_number_of_days = random.randrange(days_between_dates)
            random_date = start_date + datetime.timedelta(days=random_number_of_days)
            #create a condition
            date=str(random_date.year)+"-"+str(random_date.month)+"-"+str(random_date.day)
            conditions.append({"id":"PC"+str(i),"diagnosed":date,"cured":date,"kind":id_condition})

        dataset.append(["A"+str(j),names.get_full_name(),conditions])
        dataset_JSON=pd.DataFrame(dataset,columns=["id","name","conditions"])
        dataset_JSON.to_json("./dataset/patients.json",orient="records")

create_patients()
