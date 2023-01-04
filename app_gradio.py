# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 18:57:20 2023

@author: pauli
"""

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score, precision_score,recall_score
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import gradio as gr


data = pd.read_csv("C:/Users/pauli/Downloads/heart_failure_clinical_records_dataset.csv")

Features = list(('age', 'serum_creatinine','ejection_fraction'))

x = data[Features]
y = data["DEATH_EVENT"]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,shuffle=False)

smote = SMOTE(sampling_strategy = 'minority', random_state = 2)
x_SMOTE, y_SMOTE = smote.fit_resample(x_train,y_train)

svc = SVC()

svc.fit(x_SMOTE, y_SMOTE)

with open("model.pkl", "wb") as f:
    pickle.dump(svc, f)
    

def make_prediction(age,serum_creatinine,ejection_fraction):
    with open("model.pkl", "rb") as f:
        svc  = pickle.load(f)
        preds = svc.predict([[age,serum_creatinine,ejection_fraction]])
    if preds == 1:
            return "Patient is at high risk of dying from heart failure"
    return "Patient is at low risk of dying from heart failure"

 
age_input = gr.Number(label = "Enter the age of the patient")
#anaemia_input = gr.Radio(["No Anaemia","Anaemia is Present"], type = "index", label ="Does patient have Anaemia?")
#dm_input = gr.Radio(["No DM","DM is Present"], type = "index",label = "Does patient have Diabetes Mellitus (DM)?")
#cpk_input = gr.Number (label = "Enter level of CPK enzyme (mcg)")
cr_input = gr.Number(label = "Enter level of serum creatinine (mg)")
ef_input = gr.Number(label = "Enter ejection fraction (%)")
 

output = gr.Textbox(label= "Heart Failure Risk:", lines= 3)
output.style(height='30', rounded= True)



with gr.Blocks(css = ".gradio-container {background-color: #10217d} #md {width: 150%} ") as demo:
    
    gr.Markdown(value= """
                
                 # **<span style="color:white">Heart Failure Predictor</span>**
                
                """, elem_id="md")
    
    gr.Interface(make_prediction, inputs=[age_input,cr_input,ef_input], 
                       outputs=output, flagging_options=["clinical suspicion is high for heart failure but model says otherwise", 
                                                         "clinical suspicion is low for heart failure but model says otherwise"])
                                              #css = " div {background-color: red}",
                                              #title = "Heart Failure Predictor")
    gr.Markdown("""
                ## <span style="color:#d7baad">Input Examples</span>
                <span style="color:#d7baad">Click on the examples below for a demo of how the app runs.</span>
                """)
    gr.Examples(
        [[49, 1, 30], [65,2.7,30]],
        [age_input,cr_input,ef_input], output,
        make_prediction,
        cache_examples=True)
                                              
        
demo.launch()

#

# app = gr.Interface(make_prediction, inputs=[age_input,anaemia_input,cpk_input, dm_input,
#        cr_input,ef_input], 
#                    outputs=output, flagging_options=["clinical suspicion is high for heart failure but model says otherwise", 
#                                                      "clinical suspicion is low for heart failure but model says otherwise"],
#                    title = "Heart Failure Predictor",
#                    css="div {background-color: red}")
# app.launch()