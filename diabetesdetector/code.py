import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from openai import OpenAI

df=pd.read_csv(r'E:\11-4-24\Desktop\diabetesdetector\diabetes.csv')
print(df.columns)
st.title("Diabetes Predictor")

st.subheader("Training Data")
st.write(df.describe())

st.subheader("Visualization")
st.bar_chart(df)

x=df.drop(['Outcome'],axis=1)
y=df.iloc[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=0)

def user_report():
    Pregnancies = st.sidebar.slider('Pregnancies',0,17,3)
    Glucose=st.sidebar.slider('Glucose',0,200,120)
    BloodPressure=st.sidebar.slider("Blood Pressure",0,122,70)
    SkinThickness=st.sidebar.slider("Skin Thickness",0,100,20)
    Insulin=st.sidebar.slider("Insulin",0,846,79)
    BMI=st.sidebar.slider("BMI",0,67,20)
    DiabetesPedigreeFunction=st.sidebar.slider("Diabetes Pedigree Function",0.0,2.4,0.47)
    Age=st.sidebar.slider("Age",21,88,33)

    user_report = {
        'Pregnancies':Pregnancies,
        'Glucose':Glucose,
        'BloodPressure':BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }

    report_data = pd.DataFrame(user_report, index=[0])
    return report_data

user_data = user_report()

rf = RandomForestClassifier()
rf.fit(x_train,y_train)

st.subheader("Accuracy: ")
st.write(str(accuracy_score(y_test,rf.predict(x_test))*100)+'%')

user_result = rf.predict(user_data)
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
    output = 'You are healthy'
else:
    output = "You are not healthy"

st.write(output)



with st.expander("Common Questions About Diabetes"):
    st.markdown("""
    **Q: What are the symptoms of diabetes?**  
    A: Frequent urination, fatigue, blurry vision, and increased thirst.

    **Q: How can I prevent diabetes?**  
    A: Maintain a healthy diet, exercise regularly, and avoid excessive sugar intake.

    **Q: What is BMI?**  
    A: Body Mass Index is a measure of body fat based on height and weight.

    **Q: Is this app a medical diagnosis tool?**  
    A: No. It gives predictions based on data, but does not replace a doctor.
    """)

if user_result[0]==0:
    st.success("✅ You are healthy!")
    st.markdown("**Tip:** Keep monitoring your health and maintain a healthy lifestyle.")
else:
    st.error("⚠️ You are at risk of diabetes.")
    st.markdown("**Recommended Actions:**")
    st.markdown("- Consult a healthcare provider.\n- Reduce sugar intake.\n- Get regular exercise.\n- Monitor your glucose levels.")


import io

report_str = f"""
Diabetes Prediction Report
--------------------------
Prediction: {'At Risk' if user_result[0] == 1 else 'Healthy'}
Input Data:
{user_data.to_string(index=False)}
"""

st.download_button("📄 Download Report", report_str, file_name="diabetes_report.txt")
