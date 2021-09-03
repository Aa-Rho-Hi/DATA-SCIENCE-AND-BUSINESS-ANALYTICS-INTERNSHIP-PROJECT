#!/usr/bin/env python
# coding: utf-8

# ## GRIP @ THE SPARKS FOUNDATION¶ DATA SCIENCE AND BUSINESS ANALYTICS INTERNSHIP (SEPT21)
# 
# ## Author : Aarohi Mohrir
# 
# ## TASK 1 : Prediction using Supervised ML (Level - Beginner)
# 
# ## Linear Regression with Python Scikit Learn

# In[18]:


#importing libraries that we will be using
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #plotting library for the Python programming and its numerical mathematics extension NumPy|
import sklearn.linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[19]:


#importing the dataset
source_data = "http://bit.ly/w-data"
data = pd.read_csv(source_data)
print("Data  has been imported successfully")
data.head(25)


# In[20]:


data.shape


# Step 2 : Let's Visualize and Analyze the given Dataset¶

# In[21]:


# now plotting the dataset
data.plot(x = "Hours", y = "Scores", style = "o") 
plt.title("No of study Hours vs The percentage Scored")
plt.xlabel("No of hours of study done")
plt.ylabel("The percentage score achieved")
plt.show()


# Step 3 : Prepare the data

# In[22]:


#Extracting x and y values from the dataset
X = data.iloc[:, :-1].values  
Y = data.iloc[:, 1].values


# In[23]:


#Splitting data into training and testing set.For splitting we need to import the train_test_split() method from sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state = 0)


# In[24]:


Y_train


# In[25]:


X_test


# Step 4 :Training the algorithm

# In[26]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, Y_train) 

print("Training process completed.")


# In[27]:



# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, Y)
plt.plot(X, line);
plt.show()


# Step 6 : Predictions

# In[28]:


print(X_test) # Testing data - In Hours
Y_pred = regressor.predict(X_test) # Predicting the scores


# In[29]:


# Comparing Actual Data vs Predicted Data
df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
df


# In[30]:


# Making some own predictions using own data 
hours = 9.25
own_pred = regressor.predict([[hours]])
print(f"No of Hours studied for = {hours}")
print(f"Predicted Score achieved = {own_pred[0]}")


# Step 7 :Evaluating the model

# In[31]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))
print('Max Error:', metrics.max_error(Y_test, Y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_pred))


# # Thank you !
