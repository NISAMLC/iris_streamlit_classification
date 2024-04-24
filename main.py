
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# iris = pd.read_csv('iris.csv')



st.write("""
# Simple Iris Flower Prediction App

This app predicts the **iris flower** type

""")

st.sidebar.header('User Input Parameters')
# #function to recieve input from users
def user_input_features():

    sepal_length = st.sidebar.slider('Sepal Length',min_value=4.3,max_value=7.9,value=5.4)
    sepal_width =  st.sidebar.slider('Sepal Width',min_value=2.0,max_value=4.4,value=3.3)
    petal_length =  st.sidebar.slider('Petal Length',min_value=1.0,max_value=6.9,value=1.3)
    petal_width =  st.sidebar.slider('Petal Width',min_value=0.1,max_value=2.5,value=0.2)

    data = {
        "sepal_length":sepal_length,
        "sepal_width":sepal_width,
        "petal_length":petal_length,
        "petal_width":petal_width,
    }
    features = pd.DataFrame(data,index=[0])
    return features




df=user_input_features()

st.subheader('User Input Parameters')
st.write(df)


iris = datasets.load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(X,y)
prediction = clf.predict(df)
prediction_prob = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])

st.subheader('Prediction Probability')
st.write(prediction_prob)



