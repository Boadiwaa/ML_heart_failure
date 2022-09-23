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
from sklearn.metrics import confusion_matrix, accuracy_score
from mlxtend.plotting import plot_confusion_matrix
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.formula.api import ols
import plotly.graph_objs as gobj

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

log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
log_reg_pred = log_reg.predict(x_test)
log_reg_acc = accuracy_score(y_test, log_reg_pred)
accuracy_dict['log_reg'] = 100*log_reg_acc

print(Fore.GREEN + "Accuracy of Logistic Regression is : ", "{:.2f}%".format(100* log_reg_acc))

cm = confusion_matrix(y_test, log_reg_pred)
plt.figure()
plot_confusion_matrix(cm, hide_ticks=True)
plt.title("Confusion Matrix for Logistic Regression Model")
plt.yticks(range(2),["Heart Failure Absent", 
                     "Heart Failure Present"], fontsize = 14)

plt.xticks(range(2),["Heart Failure Absent", 
                     "Heart Failure Present"], fontsize = 10)

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
    accuracy_dict.update({'{}'.format(arg): 100*arg_acc})
    return print(Fore.GREEN + "Accuracy of {} is : {:.2f}%".format(arg , 100* arg_acc))

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
