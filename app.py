# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import flet as ft

def main(page):
    t = ft.Text(value="Heart Failure Predictor App", 
    style=ft.TextThemeStyle.DISPLAY_LARGE, bgcolor = "red",
    color = "white", text_align= "center")
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    
    answer1 = ft.TextField(label = "Does the patient have anaemia? Select Yes/No, please", value= "", text_align = "center")
    answer2 = ft.TextField(label = "Does the patient have diabetes? Select Yes/No, please",value = "", text_align = "center")
    #no = ft.TextField(value = "No", text_align = "center", width= 100)
    #yes = ft.TextField(value = "Yes", text_align = "center", width= 100)

    def btn_click(e):
        page.add(ft.Text("Patient's risk of heart failure is: ", size=20, 
                         bgcolor='red', color = 'white'))
        
    def no_click1(e):
        answer1.value = "No"
        page.update(answer1)
        
    def yes_click1(e):
         answer1.value = "Yes"
         page.update(answer1)
    
    def no_click2(e):
        answer2.value = "No"
        page.update(answer2)
        
    def yes_click2(e):
         answer2.value = "Yes"
         page.update(answer2)
         
    page.add(t, ft.Row(
        [
                    
        ft.TextField(
            label = "Enter the patient's age here, please"),
        answer1, ft.ElevatedButton("Yes", on_click=yes_click1),
    ft.ElevatedButton("No", on_click=no_click1),
        #border=ft.border.all(1, ft.colors.AMBER_600), 
        #border_radius = ft.border_radius.all(8),
        answer2, ft.ElevatedButton("Yes", on_click=yes_click2),
         ft.ElevatedButton("No", on_click = no_click2)]),
        ft.Row([
            ft.TextField(label = "Enter the patient's CPK enzyme level here (mcg/L), please"),
            ft.TextField(label = "Enter the patient's serum creatinine level here (mg/dL), please"),
            ft.TextField(label = "Enter the patient's ejection fraction here(%), please")]),
        ft.ElevatedButton("Show patient's heart failure risk category", on_click=btn_click),
        )
    
    pass

ft.app(port=2425,target=main)
