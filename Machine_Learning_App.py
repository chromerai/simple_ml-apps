#!/usr/bin/env python
# coding: utf-8

# # *Machine Learning App*
# 
# ### Overview :
# 
# This project is created in order to develop an understanding about the streamlit.io framework/library and use it to develop a basic Machine Learning Web APP in python .
# 
# We will use an iris flower dataset and build an app that predicts the class label of Iris flowers as being setosa , versicolor or viriginica .
# 
# In the front-end, the sidebar found on the left will accept input parameters pertaining to features (i.e. petal length, petal width, sepal length and sepal width) of Iris flowers. These features will be relayed to the back-end where the trained model will predict the class labels as a function of the input parameters. Prediction results are sent back to the front-end for display.
# 
# In the back-end, the user input parameters will be saved into a dataframe that will be used as test data. In the meantime, a classification model will be built. Finally, the model will be applied to make predictions on the user input data and return the predicted class labels as being one of three flower type: setosa, versicolor or virginica. Additionally, the prediction probability will also be provided that will allow us to discern the relative confidence in the predicted class labels.
# 
# ### Install prerequisite libraries :
# We  will be using three Python libraries namely streamlit, pandas and scikit-learn. You can install these libraries via the pip install command.

# In[ ]:





# ## *Code of the web app :*
# 

# In[24]:


import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[23]:


st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = (data)
    return features

df = pd.DataFrame(data = user_input_features(), index = [0])

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target
option = st.selectbox( 'Which classifier would you prefer ?',('RandomForestClassifier', 'KNeighborsClassifier'))

if option == 'RandomForestClassifier':
    clf = RandomForestClassifier()
else:
    clf = KNeighborsClassifier()


clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
#st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




