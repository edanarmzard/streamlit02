import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pickle

st.image("./pic/banner.png")

html_8="""
<div style="background-color:#EE9513;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>การทำนายรูปร่างของคน</h5></center>
</div>
"""

st.markdown(html_8,unsafe_allow_html=True)
st.markdown("")

dt=pd.read_csv("./data/iris.csv")
st.write(dt.head(10))
data1 = dt['sleep'].sum()
data2 = dt['exercise'].sum()
data3 = dt['eat'].sum()
data4 = dt['work'].sum()
dx=[data1,data2,data3,data4]
dx2=pd.DataFrame(dx, index=["d1", "d2", "d3", "d4"])
if st.button("แสดงการจินตทัศน์ข้อมูล"):
   st.area_chart(dx2)
   st.button("ไม่แสดงข้อมูล")
else:
    st.write("ไม่แสดงข้อมูล")

html_8="""
<div style="background-color:#EE9513;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>การทำนายข้อมูล</h5></center>
</div>
"""
st.markdown(html_8,unsafe_allow_html=True)
st.markdown("")

pt_len=st.slider("กรุณาเลือกชั่วโมง sleep")
pt_wd=st.slider("กรุณาเลือกชั่วโมง exercise")
sp_len=st.number_input("กรุณาเลือกชั่วโมง eat")
sp_wd=st.number_input("กรุณาเลือกชั่วโมง work")

if st.button("ทำนายผล"):
   loaded_model = pickle.load(open('./data/trained_model.sav', 'rb'))
   input_data =  (pt_len,pt_wd,sp_len,sp_wd)
   # changing the input_data to numpy array
   input_data_as_numpy_array = np.asarray(input_data)
    # reshape the array as we are predicting for one instance
   input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
   prediction = loaded_model.predict(input_data_reshaped)
   st.write(prediction)
   if prediction == 'fat':
        st.image("./pic/01.jpg")
   elif prediction == 'thin':
        st.image("./pic/02.jpg")
   else:
        st.image("./pic/03.jpg")
   st.button("slim")
else:
   st.write("ไม่แสดงข้อมูล")

    