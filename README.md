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
import pandas as pd

# Load dataset
data = pd.read_csv("Placement_Data.csv")

# Copy dataset
data1 = data.copy()

# Drop unnecessary columns
data1 = data1.drop(['sl_no', 'salary'], axis=1)

# Check null values
print(data1.isnull().sum())

# Check duplicates
print(data1.duplicated().sum())

# Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

# Input and Output
x = data1.iloc[:, :-1]
y = data1["status"]

# Split dataset
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

# Logistic Regression Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='liblinear')

# Train model
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

# Print results
print("Accuracy score:", accuracy)

print("\nConfusion matrix:\n")
print(confusion)

print("\nClassification Report:\n")
print(cr)

# Confusion Matrix Graph
from sklearn import metrics
import matplotlib.pyplot as plt

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion,
    display_labels=[True, False]
)

cm_display.plot()

plt.show()
```

## Output:
<img width="1305" height="403" alt="image" src="https://github.com/user-attachments/assets/552e3d60-ef96-4057-8029-3bcd1d825074" />

<img width="1284" height="604" alt="image" src="https://github.com/user-attachments/assets/06268c6a-e4a9-4a3a-a8cb-565f099aeae9" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
