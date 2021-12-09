from numpy.lib.function_base import append
import pandas as pd
import json
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

def get_user_item_sparse_matrix(df):
    list_succesful=[]
    list_therapy=[]
    list_id=[]
    for j in df.index:       
        for trial in df["trials"][j]:
            list_id.append(int(df["id"][j].split("A")[1]))
            list_succesful.append(int(trial["sucessful"].split("%")[0])/100)
            list_therapy.append(int(trial["therapy"].split("T")[1]))
    sparse_data = sparse.csr_matrix((list_succesful,(list_id,list_therapy)))
    return sparse_data,list_therapy,list_succesful
    
def compute_therapy_similarity_count(sparse_matrix, therapy_df, therapy_id):
    similarity = cosine_similarity(sparse_matrix.T, dense_output = False)
    name=""
    id_th=""
    for j in therapy_df.index:
        if(therapy_df["id"][j]=="T"+str(therapy_id)):
            name=therapy_df["name"][j]
            id_th=therapy_df["id"][j]
    no_of_similar_movies = name, similarity[therapy_id].count_nonzero()
    return no_of_similar_movies,id_th
def find_condition_succes(df_train,df_test):
    list_condition=[]
    list_success=[]
    unique_condition=[]
    kind=""
    for j in df_test.index:
        for condition in df_test["conditions"][j]:
            if(condition["cured"] is None):
                list_condition.append([condition["id"],condition["kind"]])
                kind=condition["kind"]
                unique_condition.append(kind)
        for condtion in df_test["conditions"][j]:
                if(condtion["kind"]==kind):
                    for trial in df_test["trials"][j]:
                        if(trial["condition"]==condtion["id"]):
                            list_success.append([kind,int(trial["therapy"].split("T")[1]),int(trial["sucessful"].split("%")[0])/100])

    for i in list_condition:
        for j in df_train.index:
            for condtion in df_train["conditions"][j]:
                if(condtion["kind"]==i[1]):
                    for trial in df_train["trials"][j]:
                        if(trial["condition"]==i[0]):
                            list_success.append([i[1],int(trial["therapy"].split("T")[1]),int(trial["sucessful"].split("%")[0])/100])
    return list_success,unique_condition
            
def get_avg_value(id_th,patients_df):
    num=0.0
    count=0
    for j in patients_df.index:
        for trial in patients_df["trials"][j]:
            if(trial["therapy"]==id_th):
                num+=int(trial["sucessful"].split("%")[0])/100
                count+=1
    print(id_th)
    if(count>0):
        num=num/count
    return num

def main():
    patients_df = pd.read_json('./dataset/patients.json',orient="records")
    test_cases=[]
    for j in patients_df.index:
        if(patients_df["conditions"][j][0]["cured"]==None):
            test_cases.append(patients_df.iloc[j])
    patients_df.drop([0,1,2],inplace=True)
    test_cases=pd.DataFrame(test_cases,columns=["id","name","conditions","trials"])
    list_therapy_train=[]
    list_therapy_test=[]
    train_sparse_data,list_therapy_train,list_success_train = get_user_item_sparse_matrix(patients_df)
    test_sparse_data,list_therapy_test,list_success_test = get_user_item_sparse_matrix(test_cases)
    therapy_df = pd.read_json('./dataset/therapy.json',orient="records")
    list_success,unique_condition=find_condition_succes(patients_df,test_cases)
    dataset_sucess=pd.DataFrame(list_success,columns=["condition","therapy","success"])
    # data_by_condition=dataset_sucess.groupby(["condition"]).mean()
    group_by=dataset_sucess.groupby(["condition","therapy"])["success"].apply(list)
    df1 = dataset_sucess.reset_index()
    # print(df1)
    # print(unique_condition)
    list_therapy=[]
    for i in unique_condition:
        max_therapy=""
        max_success=0.0
        for j in df1.index:
            if(df1["condition"][j]==i):
                if(max_success<df1["success"][j]):
                    max_therapy=df1["therapy"][j]
                    max_success=df1["success"][j]
        list_therapy.append([max_therapy,max_success])

    for i in range(0,len(unique_condition)):
        
        similar_movies,id_th = compute_therapy_similarity_count(train_sparse_data, therapy_df,list_therapy[i][0] )
        avg_new_th=get_avg_value(id_th,patients_df)
        if(avg_new_th>list_therapy[i][1]):
            print(avg_new_th,list_therapy[i][1])
            print("the best similar therapy for the condition"+ unique_condition[i]+" = {}".format(similar_movies)+str(id_th))
        else:
            print(avg_new_th,list_therapy[i][1])
            print("the best therapy for the condition"+ unique_condition[i]+" = {}".format(similar_movies)+str(list_therapy[i][0]))

main()


