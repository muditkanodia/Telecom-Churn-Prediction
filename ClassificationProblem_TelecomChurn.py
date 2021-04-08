#!/usr/bin/env python
# coding: utf-8

# ## Classification Problem - Telecom Churn Classification

# In[1]:


# imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[2]:


data = pd.read_csv('C:\\Users\\mudit\\Downloads\\telco-customer-churn\\data.csv')


# In[3]:


data.head()


# In[4]:


data.columns.values


# In[5]:


data.dtypes


# In[6]:


# Converting Total Charges to a numerical data type.
data.TotalCharges = pd.to_numeric(data.TotalCharges, errors='coerce')
data.dtypes


# In[7]:


data.isnull().sum()


# In[8]:


#After looking at the above output, we can say that there are 11 missing values for Total Charges.
#Let us replace remove these 11 rows from our data set
#Removing missing values 
data.dropna(inplace = True)

#Remove customer IDs from the data set
data = data.iloc[:,1:]

#Convertin the predictor variable in a binary numeric variable
data['Churn'].replace(to_replace='Yes', value=1, inplace=True)
data['Churn'].replace(to_replace='No',  value=0, inplace=True)


# In[9]:


data.head()


# In[10]:


data.gender = [1 if each == "Male" else 0 for each in data.gender]

columns_to_convert = ['Partner', 
                      'Dependents', 
                      'PhoneService', 
                      'MultipleLines',
                      'OnlineSecurity',
                      'OnlineBackup',
                      'DeviceProtection',
                      'TechSupport',
                      'StreamingTV',
                      'StreamingMovies',
                      'PaperlessBilling']

for item in columns_to_convert:
    data[item] = [1 if each == "Yes" else 0 if each == "No" else -1 for each in data[item]]
    
data.head()


# In[11]:


sns.countplot(x="Churn",data=data)


# In[12]:


sns.pairplot(data,vars = ['tenure','MonthlyCharges','TotalCharges'], hue="Churn")
#People having lower tenure and higher monthly charges are tend to churn more.


# In[13]:


#Also as you can see below; having month-to-month contract and fiber obtic 
#internet have a really huge effect on churn probability.
sns.set(style="whitegrid")
g1=sns.catplot(x="Contract", y="Churn", data=data,kind="bar")
g1.set_ylabels("Churn Probability")

g2=sns.catplot(x="InternetService", y="Churn", data=data,kind="bar")
g2.set_ylabels("Churn Probability")


# In[14]:


#Let's convert all the categorical variables into dummy variables
dummies = pd.get_dummies(data)
dummies.head()


# In[15]:


print(dummies.corr()["Churn"].sort_values(ascending = False))


# In[16]:


#Get Correlation of "Churn" with other variables:
plt.figure(figsize=(15,8))
dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')


# In[17]:


#Month to month contracts, absence of online security and tech support seem to be positively correlated with churn.
#While, tenure, two year contracts seem to be negatively correlated with churn.

#services such as Online security, streaming TV, online backup, tech support, etc. 
#without internet connection seem to be negatively related to churn


# In[18]:


dummies.columns.values


# In[19]:


#assign Class_att column as y attribute
y = dummies.Churn.values

#drop Class_att column, remain only numerical columns
new_data = dummies.drop(["Churn"],axis=1)


# In[20]:


#Normalize values to fit between 0 and 1. 
X = (new_data-np.min(new_data))/(np.max(new_data)-np.min(new_data)).values


# In[21]:


# we split the data into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[22]:


# import logistic regression
from sklearn.linear_model import LogisticRegression
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix


# In[23]:


# bulding the LogR model 
model_lr = LogisticRegression(solver='lbfgs')
model_lr.fit(X_train, Y_train)


# In[24]:


# bulding the NB model
model_nb = GaussianNB()
model_nb.fit(X_train, Y_train)


# In[25]:


print("Coefficients of the Logistic regression model")
coef = model_lr.coef_
intercept = model_lr.intercept_
print("Coef: ", coef)
print("Intercept: ", intercept)

#building the knn model
model_knn = KNeighborsClassifier(n_neighbors=9) 
model_knn.fit(X_train, Y_train)


# In[26]:


# predicting train set to calculate acuracy of LR model
predicted_classes_lr = model_lr.predict(X_train)

# predicting train set to calculate acuracy of NB model
predicted_classes_nb = model_nb.predict(X_train)

## predicting train set to calculate acuracy of KNN model
predicted_classes_knn = model_knn.predict(X_train)


# In[27]:


print("Confusion Matrix for LR model::")
conf_mat_lr = confusion_matrix(Y_train.tolist(),predicted_classes_lr)
print(conf_mat_lr)
sns.heatmap(conf_mat_lr,annot = True)
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')
plt.show()


# In[28]:


print("Confusion Matrix for NB model::")
conf_mat_nb = confusion_matrix(Y_train.tolist(),predicted_classes_nb)
print(conf_mat_nb)
sns.heatmap(conf_mat_nb,annot = True) #ann = {"ha": 'center',"va": 'center'}
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')
plt.show()


# In[29]:


print("Confusion Matrix for Knn model::")
conf_mat_knn = confusion_matrix(Y_train.tolist(),predicted_classes_knn)
print(conf_mat_knn)
sns.heatmap(conf_mat_knn,annot = True, xticklabels = ["Not Selected", "Selected"], yticklabels = ["Not Selected", "Selected"])
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')
plt.show()


# In[30]:


import matplotlib.image as mpimg
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[31]:


img=mpimg.imread('confusion-matrix.png')
imgplot = plt.imshow(img)
plt.show()


# In[32]:


# ### Decision boundary of Logistic Reg classifier
# decision_boundary(X,y,model_lr,"Decision boundary using Logistic Regression")  


# In[33]:


# calculate accuracy scores for train sets
accuracy_lr = accuracy_score(Y_train,predicted_classes_lr)
print("accuracy score (train) for LR model::", accuracy_lr)

accuracy_nb = accuracy_score(Y_train,predicted_classes_nb)
print("accuracy score (train) for NB model::", accuracy_nb)

accuracy_knn = accuracy_score(Y_train,predicted_classes_knn)
print("accuracy score (train) for Knn model::", accuracy_knn)


# In[34]:


### Test SET

predicted_test_classes_lr = model_lr.predict(X_test)
predicted_test_classes_nb = model_nb.predict(X_test)
predicted_test_classes_knn = model_knn.predict(X_test)


# In[35]:


print("Confusion Matrix (Test set) for LR model::")
conf_mat_test_lr = confusion_matrix(Y_test.tolist(),predicted_test_classes_lr)
print(conf_mat_test_lr)

print("Confusion Matrix (Test set) for NB model::")
conf_mat_test_nb = confusion_matrix(Y_test.tolist(),predicted_test_classes_nb)
print(conf_mat_test_nb)

print("Confusion Matrix (Test set) for Knn model::")
conf_mat_test_knn = confusion_matrix(Y_test.tolist(),predicted_test_classes_knn)
print(conf_mat_test_knn)

accuracy_test_lr = accuracy_score(Y_test,predicted_test_classes_lr)
print("accuracy score - Log Reg (Test set)::", accuracy_test_lr)

accuracy_test_nb = accuracy_score(Y_test,predicted_test_classes_nb)
print("accuracy score - Naive Bayes (Test set)::", accuracy_test_nb)

accuracy_test_knn = accuracy_score(Y_test,predicted_test_classes_knn)
print("accuracy score - Knn (Test set)::", accuracy_test_knn)


# In[36]:


## ROC AUC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

plt.figure()

logit_roc_auc = roc_auc_score(Y_test, model_lr.predict_proba(X_test)[:,1])
fpr, tpr, thresholds = roc_curve(Y_test, model_lr.predict_proba(X_test)[:,1], drop_intermediate=False) #, drop_intermediate=False
plt.plot(fpr, tpr, 'b--', label='Logistic Regression (area = %0.3f)' % logit_roc_auc)

nb_roc_auc = roc_auc_score(Y_test, model_nb.predict_proba(X_test)[:,1])
fpr_nb, tpr_nb, thresholds_nb = roc_curve(Y_test, model_nb.predict_proba(X_test)[:,1], drop_intermediate=False)
plt.plot(fpr_nb, tpr_nb, 'r:', label='Naive Bayes (area = %0.3f)' % nb_roc_auc)

knn_roc_auc = roc_auc_score(Y_test, model_knn.predict_proba(X_test)[:,1])
fpr_knn, tpr_knn, thresholds_knn = roc_curve(Y_test, model_knn.predict_proba(X_test)[:,1], drop_intermediate=False)
plt.plot(fpr_knn, tpr_knn, 'y-.', label='Knn (area = %0.3f)' % knn_roc_auc)

plt.plot([0, 1], [0, 1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('ROC')
plt.show()


# In[37]:


### Finding Optimal Classification Cut-off - Logistic Regression

### Youden's Index
# Youden's Index = J-Statistic = Max_p [Sensitivity(p)+Specificity(p)-1]

tpr_fpr = pd.DataFrame( { 'tpr': tpr,'fpr': fpr,'thresholds': thresholds } )
tpr_fpr['diff'] = tpr_fpr.tpr - tpr_fpr.fpr
tpr_fpr.sort_values( 'diff', ascending = False )[0:5]
#0.24


# In[38]:


y_pred_prob = model_lr.predict_proba(X_test)
y_pred_class = list(map(lambda x: 1 if x > 0.24 else 0, y_pred_prob[:,1]))
conf_mat_test_lr2 = confusion_matrix(Y_test.tolist(), y_pred_class)
conf_mat_test_lr2

accuracy_test_lr2 = accuracy_score(Y_test, y_pred_class)
print("New accuracy score - Log Reg (Test set)::", accuracy_test_lr2)


# In[39]:


# # SVM Classification


# In[40]:


from sklearn.svm import SVC
model_svm = SVC(gamma='auto')
model_svm.fit(X_train, Y_train)
SVC(gamma='auto')


# In[41]:


## predicting train set to calculate acuracy of SVM model
predicted_classes_svm = model_svm.predict(X_train)


# In[42]:


print("Confusion Matrix for SVM model::")
conf_mat_svm = confusion_matrix(Y_train.tolist(),predicted_classes_svm)
print(conf_mat_svm)
sns.heatmap(conf_mat_lr,annot = True)
plt.xlabel('Predicted classes')
plt.ylabel('Actual classes')
plt.show()


# In[43]:


# calculate accuracy scores for train sets
accuracy_svm = accuracy_score(Y_train,predicted_classes_svm)
print("accuracy score (train) for SVM model::", accuracy_svm)


# In[44]:


### Test SET

predicted_test_classes_svm = model_svm.predict(X_test)


# In[45]:


print("Confusion Matrix (Test set) for SVM model::")
conf_mat_test_svm = confusion_matrix(Y_test.tolist(),predicted_test_classes_svm)
print(conf_mat_test_svm)

accuracy_test_svm = accuracy_score(Y_test,predicted_test_classes_svm)
print("accuracy score - SVM (Test set)::", accuracy_test_svm)


# In[46]:


print("accuracy score - Log Reg (Test set)::", accuracy_test_lr)
print("accuracy score - Naive Bayes (Test set)::", accuracy_test_nb)
print("accuracy score - SVM (Test set)::", accuracy_test_svm)
print("accuracy score - Knn (Test set)::", accuracy_test_knn)


# In[47]:


#Accuracy is highest for Logistic Regression and SVM Model

