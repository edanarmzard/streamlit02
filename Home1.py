from nbformat import write
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots


#st.image("./pic/01.png")

html_temp="""
<div style="background-color:#EE9513;padding:10px;border-radius:10px 10px 10px 10px;border-style:'solid';border-color:black">
<h4>การวิเคราะห์และออกแบบสมรรถการเรียนรู้ด้าน AI</h4>
</div>
"""
st.image('./pic/ban1.jpg')
st.markdown(html_temp,unsafe_allow_html=True)
st.header("----------------------------------------------------------")
st.subheader("สมรรถนะการเรียนรู้ทาง AI 5 ด้าน ประกอบด้วย")
ai5="""
    1.การรับรู้สภาวะแวดล้อม <br>
    2.การแทนความรู้และการให้เหตุผล <br>
    3.การเรียนรู้ของเครื่อง <br>
    4.การโต้ตอบอย่างเป็นธรรมชาติ <br>
    5.ผลกระทบของปัญญาประดิษฐ์ต่อสังคม <br>
"""
st.markdown(ai5,unsafe_allow_html=True)
st.balloons()
st.sidebar.markdown("# 🎉หน้าแรก🎉")

def radar_chart(val1,val2,val3,val4,val5):
    df=pd.DataFrame(dict(
        r=[val1,val2,val3,val4,val5],
        theta=["ด้านที่1","ด้านที่2","ด้านที่3","ด้านที่4","ด้านที่5"]))
    fig=px.line_polar(df,r='r',theta='theta',line_close=True)
    st.write(fig)

html_2="""
<div style="background-color:#EE9513;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<h4>เกณฑ์การให้คะแนนแต่ละระดับวัดได้ดังนี้</h4>
</div>
"""
st.markdown(html_2,unsafe_allow_html=True)
st.markdown("")
st.markdown("")
html_3="""
<div style="background-color:#EE9513;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<h5>ระดับประถมศึกษา 1</h5>
</div>
"""
st.markdown(html_3,unsafe_allow_html=True)
st.markdown("")
df1 = pd.DataFrame({'ด้านที่1':6,'ด้านที่2':6,'ด้านที่3':3,'ด้านที่4':3,'ด้านที่5':9,'คะแนนรวม':27},index=(0,1))
dt1=df1.head(1)
st.dataframe(dt1) 

radar_chart(6,6,3,3,9)

html_4="""
<div style="background-color:#EE9513;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<h5>ระดับประถมศึกษา 2</h5>
</div>
"""
st.markdown(html_4,unsafe_allow_html=True)
st.markdown("")
df2 = pd.DataFrame({'ด้านที่1':3,'ด้านที่2':9,'ด้านที่3':6,'ด้านที่4':6,'ด้านที่5':9,'คะแนนรวม':33},index=(0,1))
dt2=df2.head(1)
st.dataframe(dt2) 

radar_chart(3,9,6,6,9)

html_5="""
<div style="background-color:#EE9513;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<h5>ระดับประถมศึกษา 3</h5>
</div>
"""
st.markdown(html_5,unsafe_allow_html=True)
st.markdown("")
df3 = pd.DataFrame({'ด้านที่1':6,'ด้านที่2':9,'ด้านที่3':6,'ด้านที่4':6,'ด้านที่5':6,'คะแนนรวม':33},index=(0,1))
dt3=df3.head(1)
st.dataframe(dt3) 

radar_chart(6,9,6,6,6)

html_6="""
<div style="background-color:#EE9513;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<h5>ระดับประถมศึกษา 4</h5>
</div>
"""
st.markdown(html_6,unsafe_allow_html=True)
st.markdown("")
df4 = pd.DataFrame({'ด้านที่1':9,'ด้านที่2':12,'ด้านที่3':6,'ด้านที่4':9,'ด้านที่5':6,'คะแนนรวม':42},index=(0,1))
dt4=df4.head(1)
st.dataframe(dt4) 

radar_chart(9,12,6,9,6)

html_7="""
<div style="background-color:#EE9513;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<h5>ระดับประถมศึกษา 5</h5>
</div>
"""
st.markdown(html_7,unsafe_allow_html=True)
st.markdown("")
df5 = pd.DataFrame({'ด้านที่1':9,'ด้านที่2':15,'ด้านที่3':9,'ด้านที่4':9,'ด้านที่5':9,'คะแนนรวม':51},index=(0,1))
dt5=df5.head(1)
st.dataframe(dt5) 

radar_chart(9,15,9,9,9)

html_8="""
<div style="background-color:#EE9513;padding:15px;border-radius:15px 15px 15px 15px;border-style:'solid';border-color:black">
<center><h5>เกณฑ์การให้คะแนน 4</h5></center>
</div>
"""
st.markdown(html_8,unsafe_allow_html=True)
st.markdown("")
dx=pd.read_excel('./data/gen.xlsx')
st.dataframe(dx)