# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import required libraries 
2.Create dataset with input features (e.g., CGPA, IQ) and target (placement status) 
3.Split dataset into training and testing sets 
4.Initialize the Logistic Regression model 
5.Train the model using training data
```

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Lavanya R
RegisterNumber:212225230149
*/
```
```
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.array([
    [6.5, 110],
    [7.0, 120],
    [8.0, 130],
    [5.5, 100],
    [6.0, 105],
    [7.5, 125],
    [8.5, 135],
    [5.0, 95]
])

Y = np.array([0, 1, 1, 0, 0, 1, 1, 0])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

model = LogisticRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, Y_pred))

new_student = np.array([[7.2, 118]])
prediction = model.predict(new_student)

print("Placement Prediction (1=Placed, 0=Not Placed):", prediction[0])
```

## Output:
<img width="1223" height="72" alt="image" src="https://github.com/user-attachments/assets/8fd0a84d-70ab-4016-9e91-5c6ec0b08cdb" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
