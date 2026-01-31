import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Loading dataset
os.chdir("D:/Loan-Approval-Prediction-Using-KNN/")
data = pd.read_csv('loan prediction.csv')

data.info()
data.isnull().sum()
# Pre-processing
#Imputation
#Fill in with most frequent category (mode)
data['Gender']=data['Gender'].fillna(data['Gender'].mode()[0])
data['Married']=data['Married'].fillna(data['Married'].mode()[0])
data['Dependents']=data['Dependents'].fillna(data['Dependents'].mode()[0])
data['Self_Employed']=data['Self_Employed'].fillna(data['Self_Employed'].mode()[0])
data['Credit_History']=data['Credit_History'].fillna(data['Credit_History'].mode()[0])
data['Loan_Amount_Term']=data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])

#data['LoanAmount'].describe()
data['LoanAmount']=data['LoanAmount'].fillna(data['LoanAmount'].median())

data1= data.iloc[:,1:-1]       # Exclude id and status column

#OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
cat_cols = data1.select_dtypes(include=["object"]).columns.tolist()
encoder = OneHotEncoder( handle_unknown='ignore', sparse_output=False)
# Fit and transform
encoded_data = encoder.fit_transform(data1[cat_cols])
# Create a DataFrame with the new column names
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_cols))
# Combine with the rest of the data
data1 = pd.concat([data1.drop(columns=cat_cols), encoded_df], axis=1)

X = data1.values
y = data.iloc[:,-1].values

#Splitting
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

#Scaling numeric data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,0:4] = sc.fit_transform(X_train[:,0:4])     #fit_transform on X_train
X_test[:,0:4] = sc.transform(X_test[:,0:4])           #transform on X_test

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)         #k is hyperparameter
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print('Training Accuracy : {:.3f}'.format(knn.score(X_train, y_train)))
print('Testing Accuracy : {:.3f}'.format(knn.score(X_test, y_test)))



from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred,labels=('Y','N'))

print(cm)

########

#Cross Validationn

from sklearn.model_selection import cross_val_score,KFold , cross_validate            
kf = KFold( n_splits=5,shuffle=True, random_state= 42)              
accuracies = cross_val_score(knn,X_train,y_train,cv=kf)
print('{:.3f}'.format(accuracies.mean()))

# Perform cross-validation and collect both train and test scores
cv_results = cross_validate(knn, X_train,y_train, cv=kf, scoring='accuracy', return_train_score=True)

# Extract train and test scores
train_scores = cv_results['train_score']
test_scores = cv_results['test_score']
# Show individual scores and their means
print("cross validation: train accuracy:" , np.round(train_scores.mean(),2))
print("cross validation: test accuracy:" , np.round(test_scores.mean(),2))


###############
# Elbow method to get optimum value of k

neighbors=range(1,11)
k_score=[]
for n in neighbors:
    knn1=KNeighborsClassifier(n_neighbors=n)
    accuracies1=cross_val_score(knn1,X_train,y_train,cv=kf)
    k_score.append(1-accuracies1.mean())
    
k_score
np.array(k_score).round(4)

plt.figure(figsize=(8, 5))
plt.plot(neighbors, k_score, marker='o')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Error')
plt.title('KNN Error Rate vs Number of Neighbors')

# Force integer ticks on x-axis
plt.xticks(neighbors)
plt.grid(True)
plt.show()

#Model using optimized k value
knn1=KNeighborsClassifier(n_neighbors=5)
accuracies1=cross_val_score(knn1,X_train,y_train,cv=kf)
print('{:.7f}'.format(accuracies1.mean()))

#################################################














