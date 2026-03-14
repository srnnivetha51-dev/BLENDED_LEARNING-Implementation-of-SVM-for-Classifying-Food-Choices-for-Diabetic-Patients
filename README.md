# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries such as Python, Scikit-learn, and NumPy.

2. Load the food classification dataset for training and testing.

3. Perform data preprocessing such as cleaning, normalization, and feature scaling.

4. Split the dataset into training and testing sets.

5. Initialize the Support Vector Machine model.

6. Train the SVM model using the training dataset.

7. Apply hyperparameter tuning using methods like Grid Search or Random Search.

8. Optimize parameters such as kernel type, C value, and gamma.

9. Evaluate the model performance using accuracy and classification metrics.

10. Use the optimized SVM model to classify new food items

## Program:
```
/*
Program to implement SVM for food classification for diabetic patients.
/*
Program to implement the prediction of iris species using SGD Classifier.
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns

data = pd.read_csv("food_items_binary.csv")

print(data.head())
print(data.columns)

features=['Calories','Total Fat','Saturated Fat','Sugars','Dietary Fiber','Protein']
target=['class']

x=data[features]
y=data[target]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
svm=SVC()
param_grid={
    'C':[0.1,1,10,100],
    'kernel':['linear','rbf'],
    'gamma':['scale','auto']
}scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
grid_search=GridSearchCV(svm,param_grid,cv=5,scoring='accuracy')
grid_search.fit(x_train,y_train)

best_model=grid_search.best_estimator_
print("Name: S R NIVEDHITHA")
print("Reg No: 212225240102")
print("Best Parameters:",grid_search.best_params_)
y_pred=best_model.predict(x_test)

accuracy=accuracy_score(y_test,y_pred)
print("Name: S R NIVEDHITHA")
print("Reg No: 212225240102")
print("Accuracy:",accuracy)
print("Classification Report:\n",classification_report(y_test,y_pred))

conf_matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

Developed by: S R NIVEDHITHA
RegisterNumber: 212225240102
*/
Developed by:  S R NIVEDHITHA
RegisterNumber:  212225240102
*/
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![alt text](<Screenshot 2026-03-13 122503.png>)
![alt text](<Screenshot 2026-03-13 122516.png>)
![alt text](<Screenshot 2026-03-13 122533.png>)
![alt text](<Screenshot 2026-03-13 122541.png>)
## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
