# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, drop unnecessary columns, and encode categorical variables.
2.Define the features (X) and target variable (y).
3.Split the data into training and testing sets.
4.Train the logistic regression model, make predictions, and evaluate using accuracy and other

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Sathiya Murthy k
RegisterNumber:  21222510047
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

dataset = pd.read_csv("C:/Users/acer/Downloads/Placement_Data.csv")

df = dataset.copy()
df.drop(columns=['sl_no', 'salary'], inplace=True)


print("Missing values:\n", df.isnull().sum())
print("Duplicate records:", df.duplicated().sum())

encoder = LabelEncoder()
categorical_cols = [
    'gender', 'ssc_b', 'hsc_b', 'hsc_s',
    'degree_t', 'workex', 'specialisation', 'status'
]

for col in categorical_cols:
    df[col] = encoder.fit_transform(df[col])


X = df.drop('status', axis=1)
Y = df['status']

print("Input shape:", X.shape)
print("Output shape:", Y.shape)


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(solver='liblinear'))
])

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(Y_test, Y_pred))

cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:\n", cm)

print("Classification Report:\n", classification_report(Y_test, Y_pred))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Not Placed', 'Placed']
)
disp.plot()
plt.title("Placement Prediction Confusion Matrix")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)

fig, ax = plt.subplots(figsize=(7, 6))

im = ax.imshow(cm, cmap='YlGnBu')
plt.colorbar(im)

# Change text color here
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j],
                ha="center",
                va="center",
                color="black",   # <-- changed from yellow
                fontsize=14,
                fontweight="bold")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(['Not Placed', 'Placed'])
ax.set_yticklabels(['Not Placed', 'Placed'])

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")

ax.set_title("Placement Prediction Confusion Matrix")

plt.tight_layout()
plt.show()

```

## Output:

<img width="983" height="405" alt="image" src="https://github.com/user-attachments/assets/f9526dcf-2da3-492b-8a56-83937639ac4e" />


<img width="1103" height="745" alt="image" src="https://github.com/user-attachments/assets/9c48f252-7f1c-442a-be85-98abc96fce3d" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
