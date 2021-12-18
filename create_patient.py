import names
import random
import json
import datetime
import pandas as pd

def create_patients():
    dataset=[]
    f = open('./dataset/true_condition.json', encoding="utf8")
    data = json.load(f)
    f.close()
    f = open('./dataset/therapy.json', encoding="utf8")
    data_th = json.load(f)
    start_date = datetime.date(2020, 1, 1)
    end_date = datetime.date(2020, 12, 1)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days

    for j in range(1000):
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
        #creation of trial
        nb_therapy=random.randint(3,5)
        trials=[]
        list_id_taken=[]
        for i in range(nb_therapy):
            a=False
            #at least one condition need to be link to one trial
            while(a==False):
                id_condition=random.randint(0,len(list_id_condition)-1)
                if(id_condition in list_id_taken and len(list_id_taken)>=len(conditions)):
                    a=True
                if(id_condition not in list_id_taken):
                    a=True
                    list_id_taken.append(id_condition)
            date=""
            id=""
            for item in conditions:
                if(item["kind"]==list_id_condition[id_condition]):
                   date=item["diagnosed"] 
                   id=item["id"]
            id_therapy=data_th[random.randint(0,len(data_th)-1)]["id"]
            trials.append({"id":"TR"+str(i),"start":date,"end":date,"condition":id,"therapy":id_therapy,"sucessful":str(random.randint(0,100))+"%"})
        #create the 3 test cases
        if(j<3):
            conditions[0]["cured"]= None
            """""
            del_trial=""
            for trial in trials:
                if(conditions[0]["id"]==trial["condition"]):
                    del_trial=trial
            trials.remove(del_trial)
             """""
        dataset.append(["A"+str(j),names.get_full_name(),conditions,trials])
           

        dataset_JSON=pd.DataFrame(dataset,columns=["id","name","conditions","trials"])
        dataset_JSON.to_json("./dataset/patients.json",orient="records")

create_patients()
