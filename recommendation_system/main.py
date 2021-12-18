from numpy.lib.function_base import append
import pandas as pd
import json
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
#create our matrix of patient as row and therapy as collumn filled withe sucessfull
#a sparse matrix is a matrix containing a lot of 0
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
    #We use cosine similarity to measure similarity between therapy
    #return a matrice of cosine similarity as value
    similarity = cosine_similarity(sparse_matrix.T, dense_output = False)
    row,col=similarity.get_shape()
    max_value=0.0
    similarity_therapy_id=0
    #go to the line that compute the similiarity of the therapy we want to compute
    for i in range(col):
        if(similarity[therapy_id,i]>max_value and i!= therapy_id):
            max_value=similarity[therapy_id,i]
            similarity_therapy_id=i
    name=""
    id_th=""
    # print("------")
    # print(similarity[therapy_id])
    for j in therapy_df.index:
        if(therapy_df["id"][j]=="T"+str(similarity_therapy_id)):
            name=therapy_df["name"][j]
            id_th=therapy_df["id"][j]
    return name, max_value,id_th
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
    if(count>0):
        num=num/count
    return num

def main():
    #https://pub.towardsai.net/recommendation-system-in-depth-tutorial-with-python-for-netflix-using-collaborative-filtering-533ff8a0e444
    with open("./dataset/dataset.json") as jsonFile:
        data=json.load(jsonFile)
        jsonFile.close()
    data=data["Patients"]
    patients_df=pd.DataFrame(data,columns=["id","name","conditions","trials"])
    # patients_df = pd.read_json('./dataset/patients.json',orient="records")
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
    #For each condition that we need to heal, get  the therpay link to this conditions which has the avg sucess the highest.
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
        
        similar_therapy,percent_similarity,id_th = compute_therapy_similarity_count(train_sparse_data, therapy_df,list_therapy[i][0])
        print("the therapy "+str(list_therapy[i][0])+" has a similar therapy which is "+id_th+" with "+str(percent_similarity)+" percent of similarity")
        #compute the mean of success for the similar therpay
        avg_new_th=get_avg_value(id_th,patients_df)
        list_therapy[i][0]="T"+str(list_therapy[i][0])
        avg_th=get_avg_value(list_therapy[i][0],patients_df)
        if(avg_new_th>avg_th):
            print(avg_new_th,avg_th)
            print("the best similar therapy for the condition "+ unique_condition[i]+" {} with a similarity of "+str(percent_similarity)+"".format(similar_therapy,id_th))
        else:
            name=""
            for j in therapy_df.index:
                if(therapy_df["id"][j]=="T"+str(list_therapy[i][0])):
                    name=therapy_df["name"][j]
            print(avg_new_th,list_therapy[i][1])
            print("the best therapy for the condition"+ unique_condition[i]+" = {}".format(name,list_therapy[i][0]))

main()


