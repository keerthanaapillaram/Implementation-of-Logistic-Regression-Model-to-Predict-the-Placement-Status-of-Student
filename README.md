# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

### AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

### Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook

### Algorithm
Step 1: start the program

Step 2: Load and preprocess the dataset: drop irrelevant columns, handle missing values, and encode categorical variables using LabelEncoder.

Step 3: Split the data into training and test sets using train_test_split.

Step 4: Create and fit a logistic regression model to the training data.

Step 5: Predict the target variable on the test set and evaluate performance using accuracy, confusion matrix, and classification report.

Step 6:Display the confusion matrix using metrics.ConfusionMatrixDisplay and plot the results.

Step 7:End the program.

### Program:

```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Keerthana P
RegisterNumber: 212223240069
```

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

```
data=pd.read_csv('Placement_Data.csv')
data.info()
```
### Output:
![image](https://github.com/user-attachments/assets/84ff164e-88a0-4e6d-ae81-7fd9c5441c82)

```
data=data.drop(['sl_no'],axis=1)
data
```

### Output:
![image](https://github.com/user-attachments/assets/deeafeca-63fd-4ff3-8ce4-77f85626d1f9)

```
data["gender"]=data["gender"].astype('category')
data["ssc_b"]=data["ssc_b"].astype('category')
data["hsc_b"]=data["hsc_b"].astype('category')
data["hsc_s"]=data["hsc_s"].astype('category')
data["degree_t"]=data["degree_t"].astype('category')
data["workex"]=data["workex"].astype('category')
data["specialisation"]=data["specialisation"].astype('category')
data["status"]=data["status"].astype('category')
data.dtypes

```

### Output:
![image](https://github.com/user-attachments/assets/fc833185-3289-45dc-bd7c-871ef8001b9c)

```
data["gender"]=data["gender"].cat.codes
data["ssc_b"]=data["ssc_b"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes
data["hsc_s"]=data["hsc_s"].cat.codes
data
```

### Output:
![image](https://github.com/user-attachments/assets/71235237-4554-423a-a69b-b5f5b1b2ce08)

```
data=data.drop(['salary'],axis=1)
data
```

### Output:
![image](https://github.com/user-attachments/assets/58f53075-a1e4-443c-a893-cba9c1633ccd)

```
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
y
```

### Output:
![image](https://github.com/user-attachments/assets/54bbf41c-b2bd-4280-b320-47a2e4e00724)

```
x
```

### Output:
![image](https://github.com/user-attachments/assets/dbfd8bfc-4df5-4485-8705-732650a329d6)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
```

```
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=500)
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
clf.predict(x_test)
```

### Output:
![image](https://github.com/user-attachments/assets/82133c67-b545-48da-bf07-2d426b701752)

```
from sklearn.metrics import accuracy_score
acc= accuracy_score(y_test,clf.predict(x_test))
acc
```

### Output:
![image](https://github.com/user-attachments/assets/1e6b7ccb-a620-40e2-affa-1f10f397226d)

```
clf.predict([[0,87,0,95,0,2,78,2,0,0,1,0]])
```

### Output:
![image](https://github.com/user-attachments/assets/3beef500-f229-4e58-b26a-07bebebcc8fd)


### Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.


