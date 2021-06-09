# simple_ml-apps
This project is created in order to develop an understanding about the streamlit.io framework/library and use it to develop a basic Machine Learning Web APP in python .
# 
# We will use an iris flower dataset and build an app that predicts the class label of Iris flowers as being setosa , versicolor or viriginica .
# 
# In the front-end, the sidebar found on the left will accept input parameters pertaining to features (i.e. petal length, petal width, sepal length and sepal width) of Iris flowers. These features will be relayed to the back-end where the trained model will predict the class labels as a function of the input parameters. Prediction results are sent back to the front-end for display.
# 
# In the back-end, the user input parameters will be saved into a dataframe that will be used as test data. In the meantime, a classification model will be built. Finally, the model will be applied to make predictions on the user input data and return the predicted class labels as being one of three flower type: setosa, versicolor or virginica. Additionally, the prediction probability will also be provided that will allow us to discern the relative confidence in the predicted class labels.
# 
# ### Install prerequisite libraries :
# We  will be using three Python libraries namely streamlit, pandas and scikit-learn. You can install these libraries via the pip install command.
