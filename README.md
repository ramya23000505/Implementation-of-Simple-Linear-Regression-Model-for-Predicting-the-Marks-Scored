# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored
### Date:
## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Start the program.

2.Import the standard Libraries. 

3.Set variables for assigning dataset values. 

4.Import linear regression from sklearn.

5.Assign the points for representing in the graph.

6.Predict the regression for marks by using the representation of the graph.

7.Compare the graphs and hence we obtained the linear regression for the given datas.

8.Stop te program.


## Program:

Program to implement the simple linear regression model for predicting the marks scored.

Developed by: RAMYA R

RegisterNumber: 212223230169

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/admin/Downloads/student_scores.csv")
df.head()

df.tail()

# segregating data to variables
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

# splitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

# displaying predicted values
Y_pred

Y_test

# graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# graph plot for test data
plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:

![Screenshot 2024-08-22 193427](https://github.com/user-attachments/assets/ca71e6fc-56b6-47dc-ba8a-33b6054c9f33)

![Screenshot 2024-08-22 193455](https://github.com/user-attachments/assets/881d60d0-641f-4be1-9969-3fb8523f6005)

![Screenshot 2024-08-22 193545](https://github.com/user-attachments/assets/cc140d35-0789-4e0d-b986-6a3eb108cdcd)

![Screenshot 2024-08-22 193609](https://github.com/user-attachments/assets/1a3a40ab-53ee-4940-aa51-850003dd1afe)

![Screenshot 2024-08-22 193656](https://github.com/user-attachments/assets/19da4a15-7d18-4fff-a397-8d790548dd68)

![Screenshot 2024-08-22 193715](https://github.com/user-attachments/assets/5b22f0df-500a-4ce9-985d-dcb3671e7fdd)

![Screenshot 2024-08-22 194103](https://github.com/user-attachments/assets/84194f3f-a252-4526-9612-09bbfa9358ab)

![Screenshot 2024-08-22 194247](https://github.com/user-attachments/assets/a5152002-257b-4945-b08c-b12710168144)

![Screenshot 2024-08-22 194526](https://github.com/user-attachments/assets/0b54977f-e3b0-41c7-84bd-902d1550c303)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
