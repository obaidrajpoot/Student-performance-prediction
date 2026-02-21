import pandas as pd
from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle

app = FastAPI()
with open("models/model_linear.pkl",'rb') as file:
 model=pickle.load(file)


templates = Jinja2Templates(directory='templates')

@app.get("/",response_class=HTMLResponse)
def front_page(request:Request):
    return templates.TemplateResponse('student_performance.html',{'request':request,'title':'Front Page Of Student Performance'})

@app.post("/data_student_perfromance_from_user")
async def data_student_perfromance_from_user(request:Request):
    input_user =await request.json()
    mapping ={'Low':0,'Medium':1,'High':2,'No':0,'Yes':1,'Positive':1,'Negative':0,'Neutral':2,
          'Male':0,'Female':1,'Near':0,'Moderate':2,'Far':1,'High School':0,'College':1,'Postgraduate':2,
          'Public':0,'Private':1}
    input_user=list(map(lambda x:mapping.get(x,x),input_user))
    input_user = np.array(input_user)
    data=pd.DataFrame(input_user.reshape(1,-1),columns=['Hours_Studied','Attendance','Parental_Involvement','Access_to_Resources','Extracurricular_Activities','Previous_Scores','Motivation_Level','Internet_Access','Tutoring_Sessions','Family_Income','Teacher_Quality','Peer_Influence','Physical_Activity','Learning_Disabilities','Parental_Education_Level','Distance_from_Home'])
    output = model.predict(data)[0]

    return {"Output":round(output,2)}
    
