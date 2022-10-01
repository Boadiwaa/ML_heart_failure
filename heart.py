# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 14:57:20 2022

@author: Boadiwaa
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio 
import lightgbm as lgb
import plotly.figure_factory as ff

from colorama import Fore, Back, Style  #for formatting the output font of printed texts
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score,recall_score
from mlxtend.plotting import plot_confusion_matrix
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.formula.api import ols
import plotly.graph_objs as gobj
 
from imblearn.over_sampling import SMOTE
import xgboost
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

data = pd.read_csv("C:/Users/pauli/Downloads/heart_failure_clinical_records_dataset.csv")

# Exploratory data analysis of outcome variable and age distribution
print(data.head())
print(data.shape)
print(data.columns)
print(data.info())
print(data.describe())

#We realize that the age range of the respondents is 40 to 95 years with an average of 60 years.


df = data
pio.renderers.default = "svg" #this option would allow plotly figures show inline in spyder

outcome = px.histogram(df, x= "DEATH_EVENT", text_auto=True)
outcome.update_layout(bargap=0.5, title_text= "Count of Outcomes")
outcome.show()

hist_data = [data["age"].values]
group_labels = ['age']

agedist = ff.create_distplot(hist_data, group_labels,show_rug=False, colors=['#37AA9C'])
agedist.update_layout(title_text = 'Age Distribution of Dataset')

agedist.show()


# Data Modelling with feature selection solely based on domain knowledge

Features = ['age','anaemia','creatinine_phosphokinase', 'diabetes',
            'ejection_fraction', 'serum_sodium', 'high_blood_pressure','smoking']

x = data[Features]
y = data["DEATH_EVENT"]
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=2)

accuracy_dict = {}
precision_dict = {}
recall_dict = {}

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
log_reg_pred = log_reg.predict(x_test)


cm = confusion_matrix(y_test, log_reg_pred)
plt.figure()
plot_confusion_matrix(cm, hide_ticks=True)
plt.title("Confusion Matrix for Logistic Regression Model")
plt.yticks(range(2),["Heart Failure Absent", 
                     "Heart Failure Present"], fontsize = 14)

plt.xticks(range(2),["Heart Failure Absent", 
                     "Heart Failure Present"], fontsize = 10)
print(classification_report(y_test,log_reg_pred))
print(Back.WHITE + Fore.RED + "The logistic regression model has a 23.5% sensitivity and a specificity of 95.3%. It is able to detect those who do not have the condition, better than it detects those who have the condition.")


def model(arg):
    """ Use the specified machine learning model "arg" on the data above, check for its accuracy score, append the
    accuracy score to an existing accuracy list, and return a statement shwoing the accuracy of the model in percentage form.
    
    Args:
        arg: a variable that holds an instance of the function call of a specified machine learning model.
        
    Returns:
        A printed statement stating the accuracy of the machine learning model in arg
        
    """
    arg.fit(x_train, y_train)
    arg_pred = arg.predict(x_test)
    arg_acc = accuracy_score(y_test, arg_pred)
    arg_prec = precision_score(y_test,arg_pred)
    arg_rec = recall_score(y_test,arg_pred)
    accuracy_dict.update({'{}'.format(arg): 100*arg_acc})
    precision_dict.update({'{}'.format(arg): 100*arg_prec})
    recall_dict.update({'{}'.format(arg): 100*arg_rec})
    
    pass

#Logistic Regression

log_reg = LogisticRegression()
model(log_reg)

#Support Vector Machine
svc = SVC()
model(svc)

#K Neighbours Classifier

knn = KNeighborsClassifier()
model(knn)

#Decision Tree Classifier
dt = DecisionTreeClassifier()
model(dt) 

#RandomForestClassifier
r = RandomForestClassifier()
model(r)

#GradientBoostingClassifier
gb = GradientBoostingClassifier()
model(gb)

#xgbrf (Extreme Gradient Boosting Random Forest Ensemble classifier
xgb = xgboost.XGBRFClassifier()
model(xgb)

#LGBM Classifier
lgbm = lgb.LGBMClassifier()
model(lgbm)

#Cat Boost Classifier
cat = CatBoostClassifier()
model(cat)

print(accuracy_dict)

#SMOTE: Synthetic Minority Oversampling Technique
#SMOTE is an oversampling technique where the synthetic samples are generated for the minority class. 
#This algorithm helps to overcome the overfitting problem posed by random oversampling.

#Visualization of training data before sampling technique
traindata = px.histogram(y_train, x= "DEATH_EVENT", text_auto=True)
traindata.update_layout(bargap=0.5, title_text= "Count of Outcomes for Training Dataset")
traindata.show()

#SMOTE
smote = SMOTE(sampling_strategy = 'minority')
x_train_SMOTE, y_train_SMOTE = smote.fit_resample(x_train,y_train)

#Visualization of training data after sampling technique
traindata_sampled = px.histogram(y_train_SMOTE, x= "DEATH_EVENT", text_auto=True)
traindata_sampled.update_layout(bargap=0.5, title_text= "Count of Outcomes for Training Dataset after Sampling")
traindata_sampled.show()

accuracy_dict_smote = {}
precision_dict_smote = {}
recall_dict_smote = {}

def model_2(arg):
    """ A modified version of the function "model" described above, Change is with using the training data 
    sampled with SMOTE instead of the original training data.
    Args:
        arg: a variable that holds an instance of the function call of a specified machine learning model.
        
    Returns:
        A printed statement stating the accuracy of the machine learning model in arg
        
    """
    
    arg.fit(x_train_SMOTE, y_train_SMOTE)
    arg_pred = arg.predict(x_test)
    arg_acc = accuracy_score(y_test, arg_pred)
    arg_prec = precision_score(y_test,arg_pred)
    arg_rec = recall_score(y_test,arg_pred)
    accuracy_dict_smote.update({'{}'.format(arg): 100*arg_acc})
    precision_dict_smote.update({'{}'.format(arg): 100*arg_prec})
    recall_dict_smote.update({'{}'.format(arg): 100*arg_rec})
    
    pass
   

#Logistic Regression
log_reg_2 = LogisticRegression()
model_2(log_reg_2)

#Support Vector Machine
svc_2 = SVC()
model_2(svc_2)

#K Neighbours Classifier

knn_2 = KNeighborsClassifier()
model_2(knn_2)

#Decision Tree Classifier
dt_2 = DecisionTreeClassifier()
model_2(dt_2) 

#RandomForestClassifier
r_2 = RandomForestClassifier()
model_2(r_2)

#GradientBoostingClassifier
gb_2 = GradientBoostingClassifier()
model_2(gb_2)

#xgbrf (Extreme Gradient Boosting Random Forest Ensemble classifier
xgb_2 = xgboost.XGBRFClassifier()
model_2(xgb_2)

#LGBM Classifier
lgbm_2 = lgb.LGBMClassifier()
model_2(lgbm_2)

#Cat Boost Classifier
cat_2 = CatBoostClassifier()
model_2(cat_2)

print("Accuracy dictionary for original training data: ", accuracy_dict)
print("Precision dictionary for original training data: ", precision_dict)
print("Recall dictionary for original training data: ", recall_dict)
print("Accuracy dictionary for training data after SMOTE: ", accuracy_dict_smote)
print("Precision dictionary for training data after SMOTE: ", precision_dict_smote)
print("Recall dictionary for training data after SMOTE: ", recall_dict_smote)

#Next steps:compare the model characteristics (accuracy,precision,recall) when imbalance is corrected vs the previous versions via bar graphs

#The following is just to create a dictionary of each model as the key and its metrics before and after SMOTE as the values
indices = [0,1,2,3,4,5,6,7,8]
dictlist = [accuracy_dict,precision_dict,recall_dict,accuracy_dict_smote,precision_dict_smote,recall_dict_smote]

models = list(accuracy_dict.keys())

val = []
for item in dictlist:
    val.append(list(item.values()))

metrics = [[],[],[],[],[],[],[],[],[]]

for ind in indices:
    for it in val:
        metrics[ind].append(it[ind])
        
model_dict = dict(zip(models,metrics)) 
print(model_dict) #the metrics are in the order: accuracy, precision and recall. The first three items in the list of values
#are the metrics before SMOTE, the last three are the metrics after SMOTE.


#converting the dictionary into a dataframe for better visual exploration of models and their metrics. 
df= pd.DataFrame.from_dict(model_dict, orient = 'index', columns= ["Accuracy with Raw Model","Precision with Raw Model","Recall with Raw Model",
                                                                   "Accuracy after SMOTE","Precision after SMOTE","Recall after SMOTE"])
df
