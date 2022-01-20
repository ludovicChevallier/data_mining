Before running the programm you must have these libraries:
surprise(https://pypi.org/project/scikit-surprise/)
numpy
pandas
json
After installing this libraries open with a source-code editor the folder data_mining.
In the folder recommendation_system you can:
(1)If you want to try the the recommendation system run the SVD_Recommendation.py with this command:
Python .\recommendation_system\SVD_Recommendation.py dataset.json patientID condID (from the folder data_mining)
(2)If you want to teste the RMSE for a model you can run KNN_gridSearch.py or SVD_gridSearch.py
(3) if you want to test top K with the dataset provided run test_recommendation.py
(4) If you want to do with my dataset test_recommendation_owndb.py
(5) IF you want to look at how i computed the sparsity of the matrix test_analysis.py
In the folder dataset:
(1) You can find the csv file Evaluation which contains all the test done during the process.