# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:12:49 2022

@author: prit patel
"""

import pandas as pd, numpy as np
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score, precision_score, recall_score
import warnings
import pickle
warnings.filterwarnings("ignore")


data_grp3 = pd.read_csv(r"C:\Users\user\Downloads\KSI.csv")
data_grp3

# There are several columns consist of "Yes" and "<Null>" (where Null means No). 
# For these binary column, replace  "<Null>" with"No"
unwanted = ['CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK','TRSN_CITY_VEH','EMERG_VEH','SPEEDING','REDLIGHT','ALCOHOL','DISABILITY','PASSENGER','AG_DRIV','PEDESTRIAN']
data_grp3[unwanted]=data_grp3[unwanted].replace({'<Null>':'No', 'Yes':'Yes'})

# Replace other '<Null>' with nan, printing percentage of missing values for each feature
data_grp3.replace('<Null>', np.nan, inplace=True)
data_grp3.replace(' ',np.nan,inplace=True)
print(data_grp3.isna().sum()/len(data_grp3)*100)

# Dropping columns where missing values were greater than 80%
drop_column = ['OFFSET','FATAL_NO','PEDTYPE','PEDACT','PEDCOND','CYCLISTYPE','CYCACT','CYCCOND']
data_grp3.drop(drop_column, axis=1, inplace=True)
#Drop irrelevant columns which are unique identifier
data_grp3.drop(['ObjectId','INDEX_'], axis=1, inplace=True)


print(data_grp3.shape)
print(data_grp3.isna().sum()/len(data_grp3)*100)
print(data_grp3.info())
print(data_grp3.select_dtypes(["object"]).columns)

# Neighbourhood is identical with Hood ID
data_grp3.rename(columns={'Hood ID': 'Neighbourhood'}, inplace=True)
print(data_grp3.select_dtypes(["object"]).columns)

# extract features: weekday,day, month 
data_grp3['DATE'] = pd.to_datetime(data_grp3['DATE'])
data_grp3['WEEKDAY'] =data_grp3['DATE'].dt.dayofweek
data_grp3['DAY'] = pd.to_datetime(data_grp3['DATE']).dt.day
data_grp3['MONTH'] = data_grp3['DATE'].dt.month

#Drop Date
data_grp3.drop(['DATE'], axis=1, inplace=True)
data_grp3.columns

# Neighbourhood is identical with Hood ID, drop Neighbourhood
# X,Y are longitude and latitudes, dulicate, drop X and Y
data_grp3.drop(['NEIGHBOURHOOD','X','Y'], axis=1, inplace=True)
data_grp3.columns

data_grp3['STREET1'].value_counts()
data_grp3['POLICE_DIVISION'].value_counts() 
# remove other irrelevant columns or columns contain too many missing values
data_grp3.drop(['MANOEUVER','DRIVACT','DRIVCOND','INITDIR','STREET1','STREET2','WARDNUM','POLICE_DIVISION','DIVISION'], axis=1, inplace=True)
data_grp3.columns
data_grp3.info()

#Injury
ax=sns.catplot(x='INJURY', kind='count', data=data_grp3,  hue='ACCLASS')
ax.set_xticklabels(rotation=90)
plt.title("INJURY")
data_grp3['INJURY'].value_counts()



#Visualization

#Number of Unique accidents by Year
Num_accident = data_grp3.groupby('YEAR')['ACCNUM'].nunique()
plt.figure(figsize=(12,6))
plt.title("years")
plt.ylabel('Accidents Numbers')
ax = plt.gca()
ax.tick_params(axis='x', colors='blue')
ax.tick_params(axis='y', colors='red')
my_colors = list('red')   #red, green, blue, black, etc.
Num_accident.plot(
    kind='barh', 
    color='black',
    edgecolor='red'
)
plt.show()

#Check the relation between features and target
ax=sns.catplot(x='YEAR', kind='count', data=data_grp3,  hue='ACCLASS')
ax.set_xticklabels(rotation=45)
plt.title("Accidents in different years")

#Neighborhood
ax=sns.catplot(x='DISTRICT', kind='count', data=data_grp3,  hue='ACCLASS')
ax.set_xticklabels(rotation=45)
plt.title("Accidents in different day of a week")

#Vehicle type
ax=sns.catplot(x='VEHTYPE', kind='count', data=data_grp3,  hue='ACCLASS')
ax.set_xticklabels(rotation=90)
plt.title("Vehicle type vs. occurance of accidents")

#LOCCOORD
ax=sns.catplot(x='LOCCOORD', kind='count', data=data_grp3,  hue='ACCLASS')
ax.set_xticklabels(rotation=90)
plt.title("Location Coordinate")

#INVAGE
ax=sns.catplot(x='INVAGE', kind='count', data=data_grp3,  hue='ACCLASS')
ax.set_xticklabels(rotation=90)
plt.title("Age of Involved Party")


# accident location 
#2D histogram

data_Fatal = data_grp3[data_grp3['ACCLASS'] == 'Fatal']
plt.hist2d(data_Fatal['LATITUDE'], data_Fatal['LONGITUDE'], bins=(40, 40), cmap=plt.cm.jet)
plt.title("2D histogram of fatal accidents")
plt.xlabel("LATITUDE")
plt.ylabel("LONGITUDE")
plt.show()

#scatter plot of fatal accidents
sns.scatterplot(x='LATITUDE', y='LONGITUDE', data = data_grp3[data_grp3['ACCLASS'] == 'Fatal'],alpha=0.3)
plt.title("Fatal Accidents")
plt.show()

#Data Cleaning

print(data_grp3.isna().sum()/len(data_grp3)*100)

#catagorical feature, not make much sense if impute, so keep the features, just discard these rows with missing values
data_grp3.dropna(subset=['ROAD_CLASS', 'DISTRICT','VISIBILITY','RDSFCOND','LOCCOORD','IMPACTYPE','TRAFFCTL','INVTYPE'],inplace=True)
data_grp3.isnull().sum()

#target class
data_grp3['ACCLASS']=data_grp3['ACCLASS'].replace({'Non-Fatal':0, 'Fatal':1})
data_grp3['ACCLASS'].value_counts()  
#Changing the property damage and non-fatal columns to Non-Fatal¶
data_grp3['ACCLASS'] = np.where(data_grp3['ACCLASS'] == 'Property Damage Only', 'Non-Fatal', data_grp3['ACCLASS'])
data_grp3['ACCLASS'] = np.where(data_grp3['ACCLASS'] == 'Non-Fatal Injury', 'Non-Fatal', data_grp3['ACCLASS'])

data_grp3['ACCLASS'].unique()

data_grp3['ACCLASS']=data_grp3['ACCLASS'].replace({'Non-Fatal':0, 'Fatal':1})
data_grp3['ACCLASS'].value_counts()  


#Resampling- Upsampled
from sklearn.utils import resample
dataframe=data_grp3
df_majority = dataframe[dataframe.ACCLASS==0]
df_minority = dataframe[dataframe.ACCLASS==1]
df_majority, df_minority
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=14029,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
print(df_upsampled.ACCLASS.value_counts())

data_grp3=df_upsampled


#Test Train split
#Since the dataset is unbalanced, use straified split
X = data_grp3.drop(["ACCLASS"], axis=1)
y= data_grp3["ACCLASS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5,stratify=y)
X_train, X_test, y_train, y_test
#impute
from sklearn.impute import SimpleImputer    
imputer = SimpleImputer(strategy="constant",fill_value='missing')  
data_tr=imputer.fit_transform(X_train)
data_tr= pd.DataFrame(data_tr, columns=X_train.columns)

print(data_tr.isna().sum()/len(data_tr)*100)

#numerical features
df1=data_grp3.drop(['ACCLASS'],axis=1)
num_data=df1.select_dtypes(include=[np.number]).columns
print(num_data)
data_num =data_tr[num_data] 
#standardize 
scaler = StandardScaler() #define the instance
scaled =scaler.fit_transform(data_num)
data_num_scaled= pd.DataFrame(scaled, columns=num_data)
print(data_num_scaled)

#categorical features
cat_data=df1.select_dtypes(exclude=[np.number]).columns
print(cat_data)
categoricalData =data_tr[cat_data]
print(categoricalData)

data_cat = pd.get_dummies(categoricalData, columns=cat_data, drop_first=True)
data_cat

X_train_prepared=pd.concat([data_num_scaled, data_cat], axis=1)
X_train_prepared.head()
X_train_prepared.columns
X_train_prepared.info()

import joblib
joblib.dump(X_train_prepared, r'E:\Downloads\Final_Project - Mark 2/X_train_prepared.pkl')
print("X_train_prepared dumped!")

import joblib
joblib.dump(y_train, r'E:\Downloads\Final_Project - Mark 2/y_train.pkl')
print("y_train dumped!")
import joblib
joblib.dump(y_test, r'E:\Downloads\Final_Project - Mark 2/y_test.pkl')
print("y_test dumped!")

"""#Feature Selection

#Feature selection by Logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
sel = SelectFromModel(LogisticRegression(solver='saga',penalty='l1'))
sel.fit(X_train_prepared, y_train)
selected_feat= X_train_prepared.columns[(sel.get_support())]
len(selected_feat)
print(selected_feat)


coefficient= pd.Series(sel.estimator_.coef_[0], index=X_train_prepared.columns)
#plot the selected features
fig = plt.gcf()
fig.set_size_inches(10, 20)
coefficient.plot(kind='barh')
plt.title("L1 coefficient")
plt.show()

abs_coefficient =abs(coefficient)
print(coefficient[coefficient==0])
print(coefficient[coefficient<0])
a = coefficient[coefficient>0]
print(a)

fig = plt.gcf()
fig.set_size_inches(10, 20)
a.plot(kind='barh')
plt.title("L1 coefficient")
plt.show()"""


#selected features

#numerical features
num_data=['ACCNUM', 'YEAR', 'TIME', 'HOUR', 'LATITUDE', 'LONGITUDE', 'WEEKDAY', 'DAY', 'MONTH']
data_num =data_tr[num_data] 
num_data=data_num.columns
print(num_data)

#categorical features

cat_data=['CYCLIST','AUTOMOBILE','MOTORCYCLE','TRUCK','TRSN_CITY_VEH','EMERG_VEH', 'VEHTYPE','SPEEDING','REDLIGHT','ALCOHOL','DISABILITY','PASSENGER','AG_DRIV','PEDESTRIAN',
              'ROAD_CLASS', 'DISTRICT',  'TRAFFCTL','VISIBILITY', 'LIGHT','INVTYPE', 'ACCLOC', 'LOCCOORD', 'RDSFCOND','IMPACTYPE', 'INVAGE']
categoricalData =data_tr[cat_data]
print(categoricalData.columns)
data_cat = pd.get_dummies(categoricalData, columns=cat_data, drop_first=True)
data_cat

df=pd.concat([data_num, data_cat], axis=1)
df


# Pipelines


# build a pipeline for preprocessing the categorical attributes
cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="constant",fill_value='missing')),
        ('one_hot', OneHotEncoder(drop='first')),
    ])
# build a pipeline for preprocessing the numerical attributes
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
#full transformation Column Transformer
num_attribs = num_data
cat_attribs = cat_data

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs),
    ])

X_train.dtypes
# Model Training,Tuning and Testing


#SVM

# Before Tunning 
from sklearn.svm import SVC
clf=SVC()
X_train_prepared = full_pipeline.fit_transform(X_train)
clf.fit(X_train_prepared, y_train)
#accuracy on training dataset
print("Training Accuracy",clf.score(X_train_prepared,y_train))

#test
X_test_prepared = full_pipeline.transform(X_test)
#predict
y_test_pred=clf.predict(X_test_prepared)
y_test_pred
print("accuracy of testing", accuracy_score(y_test, y_test_pred))
print("precison",precision_score(y_test, y_test_pred))
print("recall",recall_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

# After Tuning 
X_train_prepared = full_pipeline.fit_transform(X_train)

#SVM
from sklearn.svm import SVC
clf_svm=SVC()

#Random search
param_svm = [
    {'kernel': ['poly','rbf'], 
     'C': [0.01,0.1, 1],
     'gamma': [0.01, 0.05, 0.1],
     'degree':[2,3]}

  ]

random_search_svm = RandomizedSearchCV(estimator=clf_svm, param_distributions=param_svm, cv=3, scoring='accuracy', refit = True, verbose = 3)
random_search_svm.fit(X_train_prepared, y_train)
#Best parameters
print(random_search_svm.best_params_)
print(random_search_svm.best_estimator_)

random_search_svm.cv_results_

best_model= random_search_svm.best_estimator_

X_test_prepared = full_pipeline.transform(X_test)
#predict using the best model
y_test_pred = best_model.predict(X_test_prepared)

from sklearn.metrics import accuracy_score
print("After Tuning:")
print("Accuracy", accuracy_score(y_test, y_test_pred))
print("Precision", precision_score(y_test, y_test_pred))
print("Recall", recall_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))
               
svm_fullpipe_pratik= Pipeline([
        ('preprocess',full_pipeline),
        ('svc', best_model)])

import joblib
joblib.dump(svm_fullpipe_pratik, r'E:\Downloads\Final_Project - Mark 2/svm.pkl')
print("Model dumped!")

import joblib
joblib.dump(y_test_pred, r'E:\Downloads\Final_Project - Mark 2/y_test_pred_svm.pkl')
print("y_test_pred_svm dumped!")




#Logistic regression
#Befoe tuning
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=17)
X_train_prepared = full_pipeline.fit_transform(X_train)
X_train_prepared.toarray()
#test
X_test_prepared = full_pipeline.transform(X_test)
X_train_prepared.shape
lr.fit(X_train_prepared,y_train)
#predict
y_test_pred=lr.predict(X_test_prepared)
print("Training Accuracy",lr.score(X_train_prepared, y_train))
print("accuracy", accuracy_score(y_test, y_test_pred))
print("precison",precision_score(y_test, y_test_pred))
print("recall",recall_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

#After tuning
from sklearn.model_selection import GridSearchCV

grid_values = {'penalty': ['l1', 'l2'], 'C':[0.01,0.1, 1],'solver':['sag','saga']}
grid_search = GridSearchCV(lr, param_grid = grid_values,scoring = 'recall',cv=5,return_train_score=True)
grid_search.fit(X_train_prepared, y_train)

#Best parameters
print('Best parameters',grid_search.best_params_)
print('Best estimator',grid_search.best_estimator_)
cvres = grid_search.cv_results_
best_logistic_model = grid_search.best_estimator_
best_pred = best_logistic_model.predict(X_test_prepared)


from sklearn.metrics import accuracy_score
print("After Tuning:")
print("Accuracy", accuracy_score(y_test, y_test_pred))
print("Precision", precision_score(y_test, y_test_pred))
print("Recall", recall_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

lr_fullpipe_prit = Pipeline([
        ('preprocess',full_pipeline),
        ('lr', best_logistic_model)])

import joblib
joblib.dump(lr_fullpipe_prit, r'E:\Downloads\Final_Project - Mark 2/lr.pkl')
print("Model dumped!")

import joblib
joblib.dump(best_pred, r'E:\Downloads\Final_Project - Mark 2/y_test_pred_lr.pkl')
print("y_test_pred_lr dumped!")



# Random forest classifier
#Before tuning
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=0)
X_train_prepared = full_pipeline.fit_transform(X_train)
X_train_prepared.toarray()
rf.fit(X_train_prepared, y_train)
X_test_prepared = full_pipeline.transform(X_test)
rf_y_pred = rf.predict(X_test_prepared)
print('Accuracy of RandomForest is:', accuracy_score(y_test, rf_y_pred))

#After tuning
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
params_grid = {
    'n_estimators': range(10, 100, 10),
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': range(1, 10),
    'min_samples_split': range(2, 10),
    'min_samples_leaf': range(1, 10),
}

rs = RandomizedSearchCV(
    rf,
    params_grid,
    n_iter=10,
    cv=10,
    scoring='accuracy',
    return_train_score=False,
    verbose=2,
    random_state=88)

search = rs.fit(X_train_prepared, y_train)

# best parameters & estimator
print("Best Params: ", search.best_params_)
print("Best estimators are: ", search.best_estimator_)

accuracy = search.best_score_ * 100
print(
    "Accuracy for training dataset with tuning is : {:.2f}%".format(accuracy))

# Training the Random Forest Classification model on the Training Set with best param
fine_tuned_model = search.best_estimator_.fit(X_train_prepared, y_train)
# Predicting the Test set results
rf_y_pred = fine_tuned_model.predict(X_test_prepared)
# predict_proba to return numpy array with two columns for a binary classification for N and P
rf_y_scores = fine_tuned_model.predict_proba(X_test_prepared)

print('Classification Report(N): \n', classification_report(y_test, rf_y_pred))
print('Confusion Matrix(N): \n', confusion_matrix(y_test, rf_y_pred))
print('Accuracy(N): \n', metrics.accuracy_score(y_test, rf_y_pred))

rf_fullpipe_Kinjal = Pipeline([
        ('preprocess',full_pipeline),
        ('Rf', fine_tuned_model)])


import joblib
joblib.dump(rf_fullpipe_Kinjal, r'E:\Downloads\Final_Project - Mark 2/Rf.pkl')
print("Model dumped!")

import joblib
joblib.dump(rf_y_pred, r'E:\Downloads\Final_Project - Mark 2/rf_y_pred_rf.pkl')
print("rf_y_pred_rf dumped!")


# Neural Networks
#before tuning
from sklearn.neural_network import MLPClassifier

clf_neural_network = MLPClassifier(
    solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
X_train_prepared = full_pipeline.fit_transform(X_train)
X_train_prepared.toarray()
print(clf_neural_network)
clf_neural_network.fit(X_train_prepared, y_train)
clf_neural_network.score(X_train_prepared, y_train)
X_test_prepared = full_pipeline.transform(X_test)
y_pred = clf_neural_network.predict(X_test_prepared)
print('Accuracy of NN is:', accuracy_score(y_test, y_pred))

#after tuning
param_grid_mlp = {
    'activation': ['identity', 'logistic'],
    'solver': ['lbfgs', 'adam'],
    'alpha': [0.001, 0.01, 0.1],
    'learning_rate': ['constant']
}
grid_search_mlp = RandomizedSearchCV(
    estimator=clf_neural_network,
    param_distributions=param_grid_mlp,
    scoring='accuracy',
    cv=2,
    n_iter=5,
    refit=True,
    verbose=3
)
# Fitting the grid search
grid_search_mlp.fit(X_train_prepared, y_train)

# Printing the best parameters & estimator
print("Best Params: ", grid_search_mlp.best_params_)
print("Best estimators are: ", grid_search_mlp.best_estimator_)

# Classification Report

# Final Predictions Neural Network
predictions_nn = grid_search_mlp.predict(X_test_prepared)
model_accuracy_nn = metrics.accuracy_score(y_test, predictions_nn)
print(model_accuracy_nn*100)
print("Classification Report of NN:-\n ",
      metrics.classification_report(y_test, predictions_nn))


NN_fullpipe_Pooja = Pipeline([
        ('preprocess',full_pipeline),
        ('Nn', grid_search_mlp)])


import joblib
joblib.dump(NN_fullpipe_Pooja, r'E:\Downloads\Final_Project - Mark 2/NN.pkl')
print("Model dumped!")

import joblib
joblib.dump(predictions_nn, r'E:\Downloads\Final_Project - Mark 2/predictions_nn.pkl')
print("predictions_nn dumped!")


# K-Nearest Neighbour
# before tuning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics
classifier = KNeighborsClassifier(n_neighbors=2)
X_train_prepared = full_pipeline.fit_transform(X_train)
X_train_prepared.toarray()
classifier.fit(X_train_prepared, y_train)
y_pred = classifier.predict(X_test_prepared)
y_scores = classifier.predict_proba(X_test_prepared)
print('Classification Report(N): \n', classification_report(y_test, y_pred))
print('Accuracy(N): \n', metrics.accuracy_score(y_test, y_pred))

# after tuning 
knn = KNeighborsClassifier()
k_range = list(range(1, 11))
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy',
                    return_train_score=False, verbose=1)

# fitting the model for grid search
grid_search = grid.fit(X_train_prepared, y_train)

# best parameters & estimator
print("Best Params: ", grid_search.best_params_)
print("Best estimators are: ", grid_search.best_estimator_)

knn_pred = grid_search.predict(X_test_prepared)

accuracy = grid_search.best_score_ * 100
print(
    "Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))


KNN_fullpipe_Meet = Pipeline([
        ('preprocess',full_pipeline),
        ('KNN', grid_search)])


import joblib
joblib.dump(KNN_fullpipe_Meet, r'E:\Downloads\Final_Project - Mark 2/KNN.pkl')
print("Model dumped!")

import joblib
joblib.dump(y_pred, r'E:\Downloads\Final_Project - Mark 2/y_pred_knn.pkl')
print("y_pred_knn dumped!")


#decision tree classifier 
from sklearn.tree import DecisionTreeClassifier
clf_tree = DecisionTreeClassifier(criterion='gini', max_depth=5)
X_train_prepared = full_pipeline.fit_transform(X_train)
X_train_prepared.toarray()
clf_tree.fit(X_train_prepared, y_train)
X_test_prepared = full_pipeline.transform(X_test)
clf_tree_pred = rf.predict(X_test_prepared)
print('Accuracy of RandomForest is:', accuracy_score(y_test, clf_tree_pred))

Dt_fullpipe_Prit = Pipeline([
        ('preprocess',full_pipeline),
        ('DT', clf_tree)])

import joblib
joblib.dump(Dt_fullpipe_Prit, r'E:\Downloads\Final_Project - Mark 2/dt.pkl')
print("Model dumped!")

import joblib
joblib.dump(clf_tree_pred, r'E:\Downloads\Final_Project - Mark 2/clf_tree_pred.pkl')
print("clf_tree_pred dumped!")


#voting 
#hard voting

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
logistic_classifier_p = LogisticRegression(max_iter=1400)
randomf_classifier_p= RandomForestClassifier()
svm_classifier_p = SVC()
decisiontree_classifier_p = DecisionTreeClassifier(max_depth=42, criterion="entropy" )
Nuralnetwork_classifier_p = MLPClassifier()
X_train_prepared = full_pipeline.fit_transform(X_train)
X_train_prepared.toarray()

#Define a voting classifier that contains all the above classifiers as estimators, set the voting to hard
hard_vote = VotingClassifier(estimators=[('lr', logistic_classifier_p), ('rf', randomf_classifier_p),
                                    ('svm', svm_classifier_p),
                                    ('decisiontree', decisiontree_classifier_p),
                                    ('Nn',Nuralnetwork_classifier_p)], voting='hard')
hard_vote = hard_vote.fit(X_train_prepared, y_train)
y_predict = hard_vote.predict(X_test_prepared[:9])
print(y_predict)

print('Hard Voting')
for classifier in [logistic_classifier_p, randomf_classifier_p, svm_classifier_p, decisiontree_classifier_p, Nuralnetwork_classifier_p]:
	classifier.fit(X_train_prepared, y_train)
	predict = classifier.predict(X_test_prepared[:9])
	print(classifier)
	print((predict, list(y_test[:9])))
    
    






