# BLENDED_LEARNING
# Implementation of Decision Tree Model for Tumor Classification

## AIM:
To implement and evaluate a Decision Tree model to classify tumors as benign or malignant using a dataset of lab test results.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load and Prepare Dataset
2. Split the Dataset
3. Train the Decision Tree Model
4. Evaluate and Visualize Results

## Program:
```
/*
Program to  implement a Decision Tree model for tumor classification.
Developed by: Anisha A
RegisterNumber: 212225220009 
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('tumor.csv')
print(data.head())
print(data.columns)
X = data.drop(columns=['Class']) 
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Name: Anisha A")
print("Register Number:21222520009")
print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test,y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

## Output:
<img width="1026" height="355" alt="Screenshot 2026-03-11 085753" src="https://github.com/user-attachments/assets/da1e11f7-2a26-48ce-a0f7-4184ed107bb8" />
<img width="746" height="89" alt="Screenshot 2026-03-11 085819" src="https://github.com/user-attachments/assets/092c165e-38a1-46a1-be04-e37fba351bff" />
<img width="780" height="289" alt="Screenshot 2026-03-11 085832" src="https://github.com/user-attachments/assets/fc164d94-c6b2-4af0-8138-6f258be0a535" />
<img width="967" height="578" alt="Screenshot 2026-03-11 085839" src="https://github.com/user-attachments/assets/f84777f7-7af4-4a65-9114-423ce6ff54c9" />



## Result:
Thus, the Decision Tree model was successfully implemented to classify tumors as benign or malignant, and the model’s performance was evaluated.
