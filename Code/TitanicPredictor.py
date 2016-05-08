
# -*- coding: utf-8 -*-

"""
Created on Tue Mar 22 23:05:06 2016

@author: John Fuini
"""
#%%
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier  #classification tree
from sklearn.cross_validation import KFold #helper that makes it easy to do cross validation
from sklearn import svm #support vector machine
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier #random forest
import matplotlib.pyplot as plt #for plotting
#from mpl_toolkits.mplot3d import Axes3D
import f1_score 
import re



data = pd.read_csv("train.csv") #import data to train
print(data.keys())


#%%

"""
From the Kaggle page
 
 VARIABLE DESCRIPTIONS:
survival        Survival
                (0 = No; 1 = Yes)
pclass          Passenger Class
                (1 = 1st; 2 = 2nd; 3 = 3rd)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard
parch           Number of Parents/Children Aboard
ticket          Ticket Number
fare            Passenger Fare
cabin           Cabin
embarked        Port of Embarkation
                (C = Cherbourg; Q = Queenstown; S = Southampton)

SPECIAL NOTES:
Pclass is a proxy for socio-economic status (SES)
 1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower

Age is in Years; Fractional if Age less than One (1)
 If the Age is Estimated, it is in the form xx.5

With respect to the family relation variables (i.e. sibsp and parch)
some relations were ignored.  The following are the definitions used
for sibsp and parch.

Sibling:  Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
Spouse:   Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
Parent:   Mother or Father of Passenger Aboard Titanic
Child:    Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic

Other family relatives excluded from this study include cousins,
nephews/nieces, aunts/uncles, and in-laws.  Some children travelled
only with a nanny, therefore parch=0 for them.  As well, some
travelled with very close friends or neighbors in a village, however,
the definitions do not support such relations.
"""

"""
 So which features do we even think have a bearing on survival?  First note that they have 11 features
 whereas our data set has 12.  PassengerID is what is missing. This will be used to denote our passengers, 
and will not be used as a feature.   It's literally just numbering the rows and has nothing to do with the titanic. OK fine.

Now Survival will be our "labels" or "tags" for learning algorythims.

Pclass should have something to do with survival, Name will not (unless you could 
gleam nationaility and thus likely atheleticism or something which is probably a stretch)
Sex and age are important.  We should include number of siblings, although it's 
not clear if this helps or hurts. Same with parents or children.
ticket number probably doesn't mean anything.   Passenger Fare might, although 
this seems very similiar to class. Cabin might be important. 
Port of embarkation might have something to do
with where they were lodged on the boat.  

"""

# So lets cut out the things we won't need.  Since our data has a dictionary structure, we can just pop things off we don't need.
# pop gives you the element designated by the key, and also removes that element.


features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"] #list of featuers we will be usuing


"""
Data Pre-processing

Lets start by looking for missing data or non numeric entries. 
"""
print("PRE-PROCESSING:")
num_passengers = len(data["PassengerId"])
print("Our training set has %s passengers." % num_passengers)


# Pclass
# checking if any data is missing
temp = data.loc[np.isnan(data["Pclass"]), "Pclass"] 
print("Number of missing elembents in 'Pclass': %s" % len(temp)) #good. None.

# Sex
# checking if any data is missing None or Null (notice since this data is not numeric, I'm using pd.isnull instead of np.isnan)
temp = data.loc[pd.isnull(data["Sex"]), "Sex"] 
print("Number of missing elembents in 'Sex': %s" % len(temp)) #
# double checking if anything other that "male or female"
temp1 = data.loc[(data["Sex"]) == "male", "Sex"] 
temp2 = data.loc[(data["Sex"]) == "female", "Sex"] 
print("There are %s males and %s females for a total of %s non-missing genders, for %s passengers" % (len(temp1), len(temp2), len(temp1) + len(temp2), num_passengers))
# yet another way to do this
print("Possible results for 'Sex' feature: %s" % data["Sex"].unique()) #to see all unique output
# ok, so nothing is empty.  But we will need to convert these two numeric values.  I will choose 0 for male, and 1 for female.
data.loc[data["Sex"] == "male", "Sex"] = 0
data.loc[data["Sex"] == "female", "Sex"] = 1
# this loc command is a good one.  Should learn it well.  Didn't have to for loop with if statements.

# Age
# checking if any data is missing
temp = data.loc[np.isnan(data["Age"]), "Age"] 
print("Number of missing elembents in 'Pclass': %s" % len(temp)) # So we are missing the age of 177 out of 891 passengers. 
# small enough to replace with the mean.
data["Age"] = data["Age"].fillna(data["Age"].median()) #replace NaN with median age
temp = data.loc[np.isnan(data["Age"]), "Age"] #double check that no more NaN
print("After replacing missing elements in Age with the mean, the number of missing elemnts in 'Age' is now %s" % len(temp))

# SibSp
# checking if any data is missing
temp = data.loc[np.isnan(data["SibSp"]), "SibSp"] 
print("Number of missing elembents in 'SibSp': %s" % len(temp)) #good. None.

# Parch
# checking if any data is missing
temp = data.loc[np.isnan(data["Parch"]), "Parch"] 
print("Number of missing elembents in 'Parch': %s" % len(temp)) #good. None.

# Fare
# checking if any data is missing
temp = data.loc[np.isnan(data["Fare"]), "Fare"] 
print("Number of missing elembents in 'Fare': %s" % len(temp)) #good. None.

# Cabin
# checking if any data is missing
temp = data.loc[pd.isnull(data["Cabin"]), "Cabin"] 
print("Number of missing elembents in 'Cabin': %s" % len(temp)) #OKAY.  Here we see 687 missing values...
# That is a huge fraction of 891.  We should strike the whole feature.  Can't  build mean out of such a small fraction and have it mean anything.
# plus how do you even "mean" the cabin numbers.  Lets get rid of this.
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

#Embarked
# checking if any data is missing, again non-numeric
print("Possible results for 'Embarked' feature: %s" % data["Embarked"].unique()) #lets see what output exists
temp = data.loc[pd.isnull(data["Embarked"]), "Embarked"] 
print("Number of missing elembents in 'Embarked': %s" % len(temp)) #two are missing.  We're gonna have to figure out what to do with them.
temp1 = data.loc[(data["Embarked"]) == "S", "Embarked"] 
temp2 = data.loc[(data["Embarked"]) == "C", "Embarked"] 
temp3 = data.loc[(data["Embarked"]) == "Q", "Embarked"] 
print("Embarked from 'S': %s, from 'C': %s, and from 'Q': %s. Grad total embarked non-empty: %s." % (len(temp1), len(temp2), len(temp3), len(temp1) + len(temp2)+ len(temp3)))
# since we have an overwhelming number of 'S', I'm going to place the two missing with 'S'. (my 'mean' so to speak)
data["Embarked"] = data["Embarked"].fillna("S")
print("Filled the two missing with 'S'")
# and now we will replace these all with 'S' = 0, 'C' = 1, 'Q' = 2
data.loc[data["Embarked"] == "S", "Embarked"] = 0
data.loc[data["Embarked"] == "C", "Embarked"] = 1
data.loc[data["Embarked"] == "Q", "Embarked"] = 2

#We have cleaned are data and we want to restrict to these features.  Also we 
# want to isolate the labels (what we are trying to predict)

Xy = data[features + ["Survived"]] #our restricted data matrix, (rows are passengers, columns are features)
original_features = list(features) # making a *COPY* of the list

#%%
"""
Feature Generation

"""


"""
Determine parent or chidren!


Here we want to split up our parent/child column into something that kind of dilineates 
dependent or caretaker 

Basically we want to make an dependet score if much lower that 20 y.o. or much greater that 60 years old,
and then we want to  multiply this by the number of caretakers/dependents you have.

We will conver the age to be negative if between 20 - 60, and positive other wise.  

So, score =(age - 20)(age - 60)*Parch
"""

#Xy["Dependence"] = Xy["Parch"]*(Xy["Age"]-20)*(Xy["Age"]-60)
#features.append("Dependence")
#  This feature seemed to make classifiers worse....


#%%
"""
Titles

why is this taking so long?
"""
Xy["Title"] = np.NaN 
titles = {}
for i in range(len(data["Name"])):
    test = re.sub('(.*, )|(\\..*)','', data.loc[i, "Name"])
    print(i)
    Xy.loc[i,"Title"] = str(test)
    if test not in titles.keys():    
        titles[test] = 1
    else:
        titles[test] = titles[test] + 1
#print(titles)        


#%%
"""


"""
Xy["TitleNum"] = 0
for i in range(len(Xy["Title"])):
    print(i)
    if Xy.loc[i, "Title"] in ["Col","Capt","Major"]:
        Xy.loc[i, "TitleNum"] = 5
    elif Xy.loc[i, "Title"] in ["Dr"]:     
        Xy.loc[i, "TitleNum"] = 4
    elif Xy.loc[i, "Title"] in ["Mr"]:     
        Xy.loc[i, "TitleNum"] = 3
    elif Xy.loc[i, "Title"] in ["Rev"]:     
        Xy.loc[i, "TitleNum"] = 2
    elif Xy.loc[i, "Title"] in ["Master", "Sir"]:     
        Xy.loc[i, "TitleNum"] = 1
    elif Xy.loc[i, "Title"] in ["Mrs", "Miss", "Ms"]:     
        Xy.loc[i, "TitleNum"] = -1
    elif Xy.loc[i, "Title"] in ["Lady"]:     
        Xy.loc[i, "TitleNum"] = -2
    else:
        Xy.loc[i, "TitleNum"] = 0

features.append("TitleNum") 
Xy = Xy[features + ["Survived"]]        
        
        
        
#%%

"""
Data Standardization

We want the variation of all of our data channels to have zero mean and unit variance. 
This just stops certain features from *looking* more important than they are.

There are some features that are basically labels, and not numbers, like "Sex" and "Embarked" and "Pclass", I wont standardize these. 
(the idea is that we'd like to have all features varying over order one numbers)

We will use the following standardization for each feature we standardize:
x -> (x - mean)/std_deviation

"""
Xy_UnNorm = Xy.copy(deep = True)

def standardize_series(series):
    return (series-series.mean())/series.std()

Xy["Sex"] = standardize_series(Xy["Sex"])
Xy["Age"] = standardize_series(Xy["Age"])
Xy["SibSp"] = standardize_series(Xy["SibSp"])
Xy["Parch"] = standardize_series(Xy["Parch"])
Xy["Fare"] = standardize_series(Xy["Fare"])
#Xy["Dependence"] = standardize_series(Xy["Dependence"])
Xy["TitleNum"] = standardize_series(Xy["TitleNum"])








#%%
"""
Data Analysis

Now with our data clean, we can start doing some machine learning on it!

"""

"""
Classificaltion Tree

"""


print("CLASSIFICATION (Classification Tree):")
X = Xy[features]
labels = Xy["Survived"]


#how trees work (note the next lines are just an example, and not needed for code)
#tree = DecisionTreeClassifier()
#tree.fit(X, labels)
#tree.predict()

#make a tree
tree = DecisionTreeClassifier()

#cross validation and error 
predictions = []
num_folds = 10
kf = KFold(num_passengers, n_folds=num_folds, random_state=1)
# kfold splits up num_passengers by randomly selecting indicides and putting them into n_folds number of arrays.  
#Then n_folds - 1 are all packed together and called 'train', 
# and the last 1 is given to 'test'.  And it returns the n_folds ways there are to do this.
# Basically if n_folds = 10, it gives me a randomly splitting into tenths, and passes me 9/10 as train and 1 tenth as test,
# and it returns all 10 ways to do this.  So one wants to train on the training sets and test on the remaining set, and do this ten times...
# for the ten different random partitionings given to us and get an average error
accuracy = np.zeros(num_folds)
f1_scores = np.zeros(num_folds)
i = 0;
for train, test in kf:    
    #setting up the partitions of our data
    X_train = X.loc[train] #subset of data to train on
    X_test = X.loc[test] #subset of data to test on
    labels_train = labels.loc[train]#labels pertaining to the train data
    labels_test = labels.loc[test] #labels pertaining to the test data
    
    #training the tree
    tree.fit(X_train, labels_train) #fitting only on the training data
    predictions = tree.predict(X_test)
    labels_test = labels_test.values #just turning this into an ndarray instead of a pandas series
    
    # monitor success rate
    failure_array = abs(predictions - labels_test) # correct survival prediction will produce 0, fails will produce 1
    total_tested = len(labels_test)
    frac_failed = sum(failure_array)/float(total_tested)
    accuracy[i] = (1 - frac_failed)
    f1_scores[i] = f1_score.f1(predictions, labels_test)
    i = i + 1 #probably a more intelligent way to do this.
    
print("TREE: Percentages of success for 10-fold cross validation")
#print(accuracy)
ave_success = sum(accuracy)/num_folds
print("Average success rate: %s" % ave_success)
print("TREE: F1 scores for 10-fold cross validation")
#print(f1_scores)
ave_f1 = sum(f1_scores)/num_folds
print("Average F1 score: %s" % ave_f1)


#%%
"""
Random Forest

"""

print("CLASSIFICATION (Random Forest):")


#make a random forest classifier
rf = RandomForestClassifier(n_estimators=50)

#cross validation and error 
predictions = []
num_folds = 10
kf = KFold(num_passengers, n_folds=num_folds, random_state=1)
accuracy = np.zeros(num_folds)
f1_scores = np.zeros(num_folds)
i = 0;
for train, test in kf:    
    #setting up the partitions of our data
    X_train = X.loc[train] #subset of data to train on
    X_test = X.loc[test] #subset of data to test on
    labels_train = labels.loc[train]#labels pertaining to the train data
    labels_test = labels.loc[test] #labels pertaining to the test data
    
    #training the tree
    rf.fit(X_train, labels_train) #fitting only on the training data
    predictions = rf.predict(X_test)
    labels_test = labels_test.values #just turning this into an ndarray instead of a pandas series
    
    # monitor success rate
    failure_array = abs(predictions - labels_test) # correct survival prediction will produce 0, fails will produce 1
    total_tested = len(labels_test)
    frac_failed = sum(failure_array)/float(total_tested)
    accuracy[i] = (1 - frac_failed)
    f1_scores[i] = f1_score.f1(predictions, labels_test)
    i = i + 1 #probably a more intelligent way to do this.
    
print("Random Forest: Percentages of success for 10-fold cross validation")
#print(accuracy)
ave_success = sum(accuracy)/num_folds
print("Average success rate: %s" % ave_success)
print("Random Forest: F1 scores for 10-fold cross validation")
#print(f1_scores)
ave_f1 = sum(f1_scores)/num_folds
print("Average F1 score: %s" % ave_f1)






#%%
"""
SVM Linear

"""
print("CLASSIFICATION (Support Vector Machines - Linear):")

#how SVMs work
#titanicSVM = svm.SVC()
#titanicSVM.fit(X, labels) # how to train an svm
#titanicSVM.predict()            # how to predict with svm

# make an SVM
titanicSVMlin = svm.SVC()

#cross validation and error for SVM (same setup as with the tree above)
predictions = []
num_folds = 10
kf = KFold(num_passengers, n_folds=num_folds, random_state=1)
accuracy = np.zeros(num_folds)
f1_scores = np.zeros(num_folds)
i = 0;
for train, test in kf:    
    #setting up the partitions of our data
    X_train = X.loc[train] #subset of data to train on
    X_test = X.loc[test] #subset of data to test on
    labels_train = labels.loc[train]#labels pertaining to the train data
    labels_test = labels.loc[test] #labels pertaining to the test data
    
    #training the SVM
    titanicSVMlin.fit(X_train, labels_train) #fitting only on the training data 
    predictions = titanicSVMlin.predict(X_test)
    labels_test = labels_test.values #just turning this into an ndarray instead of a pandas series
    
    # monitor success rate
    failure_array = abs(predictions - labels_test) # correct survival prediction will produce 0, fails will produce 1
    total_tested = len(labels_test)
    frac_failed = sum(failure_array)/float(total_tested)
    accuracy[i] = (1 - frac_failed)
    f1_scores[i] = f1_score.f1(predictions, labels_test)
    i = i + 1 #probably a more intelligent way to do this.
    

#print("SVM (linear): Percentages of success for 10-fold cross validation: %s" % accuracy)
ave_success = sum(accuracy)/num_folds
print("Average success rate for linear: %s" % ave_success)   
print("SVM (linear): F1 scores for 10-fold cross validation")
#print(f1_scores)
ave_f1 = sum(f1_scores)/num_folds
print("Average F1 score for linear: %s" % ave_f1)
print("this is a better F1 than the Tree")


#%%

"""
SVM guassian (finding Parameters)

"""

# First use a cross validation set to find the best parameters, iterate over a
# parameter grid

print("CLASSIFICATION (Support Vector Machines - Gaussian):")
#gamma_vec = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300]
#C_vec = gamma_vec
#gamma_vec = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
#C_vec = [0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2]
#gamma_vec = [0.15, 0.17, 0.19, 0.21, 0.23, 0.25]
#C_vec = [0.75, 0.77, 0.79, 0.81, 0.83, 0.85]
gamma_vec = [0.205, 0.21, 0.215, 0.22, 0.225, 0.23, 0.235]
C_vec = [0.68, 0.69, 0.70, 0.71, 0.72, 0.73, 0.74, 0.75]
num_folds = 10
param_grid = dict(gamma=gamma_vec, C=C_vec)
kf = KFold(num_passengers, n_folds=num_folds, random_state=1)
grid = GridSearchCV(svm.SVC(kernel = 'rbf'), param_grid=param_grid, cv=kf)
grid.fit(X, labels)

print("The best parameters are %s with a score of %0.4f"
      % (grid.best_params_, grid.best_score_))

#The best parameters are {'C': 0.68, 'gamma': 0.225} with a score of 0.8305
     
gamma = grid.best_params_['gamma']
C = grid.best_params_['C']     

#%%
"""
SVM guassian (building classifier)
"""
# Really now, this should be used on TEST data, and not the cross validation data.  

#how SVMs work
#titanicSVM = svm.SVC()
#titanicSVM.fit(X, labels) # how to train an svm
#titanicSVM.predict()            # how to predict with svm

# make an SVM
titanicSVMgauss = svm.SVC(kernel = "rbf", C = C, gamma = gamma)

#cross validation and error for SVM (same setup as with the tree above)
predictions = []
num_folds = 10
kf = KFold(num_passengers, n_folds=num_folds, random_state=1)
accuracy = np.zeros(num_folds)
f1_scores = np.zeros(num_folds)
i = 0;
for train, test in kf:    
    #setting up the partitions of our data
    X_train = X.loc[train] #subset of data to train on
    X_test = X.loc[test] #subset of data to test on
    labels_train = labels.loc[train]#labels pertaining to the train data
    labels_test = labels.loc[test] #labels pertaining to the test data
    
    #training the SVM
    titanicSVMgauss.fit(X_train, labels_train) #fitting only on the training data 
    predictions = titanicSVMgauss.predict(X_test)
    labels_test = labels_test.values #just turning this into an ndarray instead of a pandas series
    
    # monitor success rate
    failure_array = abs(predictions - labels_test) # correct survival prediction will produce 0, fails will produce 1
    total_tested = len(labels_test)
    frac_failed = sum(failure_array)/float(total_tested)
    accuracy[i] = (1 - frac_failed)
    f1_scores[i] = f1_score.f1(predictions, labels_test)
    i = i + 1 #probably a more intelligent way to do this.
    

#print("SVM (Gaussian): Percentages of success for 10-fold cross validation: %s" % accuracy)
ave_success = sum(accuracy)/num_folds
print("Average success rate for SVM Gaussian: %s" % ave_success)        
print("SVM (Gaussian): F1 scores for 10-fold cross validation")
#print(f1_scores)
ave_f1 = sum(f1_scores)/num_folds
print("Average F1 score for SVM Gaussian: %s" % ave_f1)
print("this is a *barely* better F1 than the SVM linear")






#%%
"""

SVM Gaussian Visualiation


"""
  
#2D classifier for visualization
Xy_2d_males = Xy_UnNorm[["Age", "SibSp" ,"Survived"]][Xy_UnNorm["Sex"]==0]
Xy_2d_females = Xy_UnNorm[["Age", "SibSp", "Survived"]][Xy_UnNorm["Sex"]==1]
X_2d_males = Xy_2d_males[["Age","SibSp"]]
X_2d_females = Xy_2d_females[["Age","SibSp"]]
y_males = Xy_2d_males["Survived"]
y_females = Xy_2d_females["Survived"]

plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(X_2d_males["Age"].min()-5, X_2d_males["Age"].max()+5, 200), 
                     np.linspace(X_2d_males["SibSp"].min()-2, X_2d_males["SibSp"].max()+2, 200))

titanicSVMgauss.fit(X_2d_males, y_males)


Z = titanicSVMgauss.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


#plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)

plt.title("gamma=%.2f, C=%.2f" % (gamma, C), size='medium')


plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
plt.scatter(X_2d_males.iloc[:, 0], X_2d_males.iloc[:, 1], c=y_males, cmap=plt.cm.RdBu_r)
plt.xticks((0,10,20,30,40,50,60,70))
plt.yticks((0,1,2,3,4,5,6,7,8))
plt.title('Males Survival Decision Boundaries')
plt.xlabel('Age')
plt.ylabel('SibSp')
plt.axis('tight')
plt.show()

plt.figure(figsize=(8, 6))
xx, yy = np.meshgrid(np.linspace(X_2d_females["Age"].min()-5, X_2d_females["Age"].max()+5, 200), 
                     np.linspace(X_2d_females["SibSp"].min()-2, X_2d_females["SibSp"].max()+2, 200))

titanicSVMgauss.fit(X_2d_females, y_females)


Z = titanicSVMgauss.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


#plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)

plt.title("gamma=%.2f, C=%.2f" % (gamma, C), size='medium')

print(gamma)
plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
plt.scatter(X_2d_females.iloc[:, 0], X_2d_females.iloc[:, 1], c=y_females, cmap=plt.cm.RdBu_r)
plt.xticks((0,10,20,30,40,50,60,70))
plt.yticks((0,1,2,3,4,5,6,7,8))
plt.title('Females Survival Decision Boundaries')
plt.xlabel('Age')
plt.ylabel('SibSp')
plt.axis('tight')
plt.show()








#%%
"""
Singular Value Decomposition

Trying to find the most important features.
(this really requires the standardization of features done awhile back)

"""

#we need to make this a np.array and not a pandas DataFrame
X_matrix = X.values
X_matrix = X_matrix[:,1:] #removing the first colum which just numbers the rows and is not a feature
X_matrix = X_matrix.astype(float) #converting the object array into an array of floats for maths

U, s, Vt = np.linalg.svd(np.transpose(X_matrix)) #perform the SVD (transposed the data to get it in a form I am familiar with)
plt.plot(s/sum(s), 'ro') #plotting the percentage of "energy" in each singular value
plt.xlabel('Principal Mode Number')
plt.ylabel('Percentage of energy')
plt.title('Singular Values')
plt.show()

#lets just check what the first two dominant modes feature composition is
plt.plot(abs(U[0]), 'ro', abs(U[1]), 'ko') #plotting the percentage of "energy" in each singular value
plt.xlabel('Principal Mode Decomposition into our features (1, 2, and 3 are "Sex", "Age", and "SibSp")')
plt.ylabel('Component of mode in each feature direction')
plt.title('Principal modes projected onto our features. ')
plt.legend(["Mode 1", "Mode 2"])
plt.show()

#One thing we see from the principle component analysis is that Sex Age and SibSP are important.  While we might guess
#that women and children are saved first, what does siblings have to do with it? Do siblings help or hurt?  Lets find out!
#need to add the survided labels to the data set.

#Xy = data[features + ["Survived"]]
sibs_ave_survival = Xy.pivot_table(values = "Survived", index = "SibSp", aggfunc = np.mean)
plt.plot(sibs_ave_survival, 'ko')
plt.title('Siblings help or hurt your survival chance?')
plt.xlabel('Number of Siblings')
plt.ylabel('Survival mean')
plt.show() #so we see that the first sibling helps you, two siblings is still better than none, but by the third it hurts you



#%%

"""
Age Breakdown

What are the survival probabilities for different age groups

We'll call 
Children < 6
6 <= Juvenile < 12
12 <= Teen < 18
else:  Adults  

"""


def age_label(row):
    if row["Age"] < 6:
        return 0 #Children
    elif row["Age"] < 12:
        return 1 #Juvenile
    elif row["Age"] < 18:
        return 2 # Teen
    else: 
        return 3 #Adult
        
Xy_UnNorm["Age_Group"] = Xy_UnNorm.apply(age_label, axis = 1)

age_groups_by_sex_survival = Xy_UnNorm.pivot_table(index = ["Age_Group", "Sex"], values = "Survived", aggfunc = np.mean)

males_survival_by_age_group = []
females_survival_by_age_group = []
for i in range(0,4):  #gosh I such at coding.  There has got to be a better way to do this
    males_survival_by_age_group.append(age_groups_by_sex_survival.iloc[2*i])
    females_survival_by_age_group.append(age_groups_by_sex_survival.iloc[2*i + 1])
    
plt.plot([0,6,12,18], males_survival_by_age_group, 'bo', [0,6,12,18], females_survival_by_age_group, 'ro' )
plt.title('Surival by age group and Sex')
plt.xlabel('Age and over')
plt.ylabel('Survival mean')
plt.show()





#%%
"""
Visualization


Lets visualize the 3 pairings of the top three important features
"""

#Xy["Survied"][Xy["Survived"]==1]

s_frame = Xy_UnNorm[["Sex", "Age", "SibSp"]][Xy_UnNorm["Survived"]==1]
d_frame = Xy_UnNorm[["Sex", "Age", "SibSp"]][Xy_UnNorm["Survived"]==0]
s_males_frame = s_frame[["Age", "SibSp"]][s_frame["Sex"]==1]
d_males_frame = d_frame[["Age", "SibSp"]][d_frame["Sex"]==1]
s_females_frame = s_frame[["Age", "SibSp"]][s_frame["Sex"]==0]
d_females_frame = d_frame[["Age", "SibSp"]][d_frame["Sex"]==0]



#s = s_frame.values.astype(float)
#d = d_frame.values.astype(float)
s_males = (s_males_frame.values)
d_males = (d_males_frame.values)
s_females = (s_females_frame.values)
d_females = (d_females_frame.values)
"""
died_scatter = died_frame[["Sex", "Age", "SibSp"]].values
survivor_scatter = survivor_scatter.astype(float)
died_scatter = died_scatter.astype(float)
"""
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(survivor_scatter[:,0], survivor_scatter[:,1], survivor_scatter[:,2], c='k')
ax.scatter(died_scatter[:,0], died_scatter[:,1], died_scatter[:,2], c='r')

plt.show()
"""


fig = plt.figure()
plt.plot(s_males[:,0], s_males[:,1], "bo")
plt.plot(d_males[:,0], d_males[:,1], "ro")
plt.title('Males')
plt.xlabel('Age')
plt.ylabel('SibSp')
plt.show()

fig = plt.figure()
plt.plot(s_females[:,0], s_females[:,1], "bo")
plt.plot(d_females[:,0], d_females[:,1], "rx")
plt.title('Females')
plt.xlabel('Age')
plt.ylabel('SibSp')
plt.show()


#%%

"""

Kaggle Submission


Our best predictor was the Guassian SVM with C and sigma chosen to be 
gamma = 0.2
C = 0.83    

"""



test = pd.read_csv('test.csv')
print(test.head())

#%%
"""

Cleaning the Test data

"""

# Pclass
# checking if any test data is missing
temp = test.loc[np.isnan(test["Pclass"]), "Pclass"] 
print("Number of missing elembents in 'Pclass': %s" % len(temp)) #good. None.

# Sex
# checking if any data is missing None or Null (notice since this data is not numeric, I'm using pd.isnull instead of np.isnan)
temp = test.loc[pd.isnull(test["Sex"]), "Sex"] 
print("Number of missing elembents in 'Sex': %s" % len(temp)) #
test.loc[test["Sex"] == "male", "Sex"] = 0
test.loc[test["Sex"] == "female", "Sex"] = 1


# Age
# checking if any data is missing
temp = test.loc[np.isnan(data["Age"]), "Age"] 
print("Number of missing elembents in 'Pclass': %s" % len(temp)) # So we are missing the age of 177 out of 891 passengers. 
# small enough to replace with the mean.
test["Age"] = test["Age"].fillna(test["Age"].median()) #replace NaN with median age
temp = test.loc[np.isnan(test["Age"]), "Age"] #double check that no more NaN
print("After replacing missing elements in Age with the mean, the number of missing elemnts in 'Age' is now %s" % len(temp))


# SibSp
# checking if any data is missing
temp = test.loc[np.isnan(test["SibSp"]), "SibSp"] 
print("Number of missing elembents in 'SibSp': %s" % len(temp)) #good. None.

# Parch
# checking if any data is missing
temp = test.loc[np.isnan(test["Parch"]), "Parch"] 
print("Number of missing elembents in 'Parch': %s" % len(temp)) #good. None.

# Fare
# checking if any data is missing
temp = test.loc[np.isnan(test["Fare"]), "Fare"] 
print("Number of missing elembents in 'Fare': %s" % len(temp)) #Just one
test["Fare"] = test["Fare"].fillna(test["Fare"].median()) 

# Cabin
# checking if any data is missing
temp = test.loc[pd.isnull(test["Cabin"]), "Cabin"] 
print("Number of missing elembents in 'Cabin': %s" % len(temp)) #OKAY.  Here we see 687 missing values...
# That is a huge fraction of 891.  We should strike the whole feature.  Can't  build mean out of such a small fraction and have it mean anything.
# plus how do you even "mean" the cabin numbers.  Lets get rid of this.

#Embarked
# checking if any data is missing, again non-numeric
print("Possible results for 'Embarked' feature: %s" % test["Embarked"].unique()) #lets see what output exists
temp = test.loc[pd.isnull(test["Embarked"]), "Embarked"] 
print("Number of missing elembents in 'Embarked': %s" % len(temp)) #two are missing.  We're gonna have to figure out what to do with them.
temp1 = test.loc[(test["Embarked"]) == "S", "Embarked"] 
temp2 = test.loc[(test["Embarked"]) == "C", "Embarked"] 
temp3 = test.loc[(test["Embarked"]) == "Q", "Embarked"] 
print("Embarked from 'S': %s, from 'C': %s, and from 'Q': %s. Grad total embarked non-empty: %s." % (len(temp1), len(temp2), len(temp3), len(temp1) + len(temp2)+ len(temp3)))
# since we have an overwhelming number of 'S', I'm going to place the two missing with 'S'. (my 'mean' so to speak)


test.loc[test["Embarked"] == "S", "Embarked"] = 0
test.loc[test["Embarked"] == "C", "Embarked"] = 1
test.loc[test["Embarked"] == "Q", "Embarked"] = 2

#We have cleaned are data and we want to restrict to these features.  Also we 
# want to isolate the labels (what we are trying to predict)

X_test = test[original_features] #our restricted data matrix, (rows are passengers, columns are features)

"""
Repeat Feature Generation/Normalization
"""
#%%
"""
Titles

why is this taking so long?
"""
X_test["Title"] = np.NaN 
titles = {}
for i in range(len(X_test["Title"])):
    temp_title = re.sub('(.*, )|(\\..*)','', data.loc[i, "Name"])
    print(i)
    X_test.loc[i,"Title"] = str(temp_title)
    if test not in titles.keys():    
        titles[temp_title] = 1
    else:
        titles[temp_title] = titles[temp_title] + 1
#print(titles)        


#%%
"""


"""
X_test["TitleNum"] = 0
for i in range(len(X_test["Title"])):
    print(i)
    if X_test.loc[i, "Title"] in ["Col","Capt","Major"]:
        X_test.loc[i, "TitleNum"] = 5
    elif X_test.loc[i, "Title"] in ["Dr"]:     
        X_test.loc[i, "TitleNum"] = 4
    elif X_test.loc[i, "Title"] in ["Mr"]:     
        X_test.loc[i, "TitleNum"] = 3
    elif X_test.loc[i, "Title"] in ["Rev"]:     
        X_test.loc[i, "TitleNum"] = 2
    elif X_test.loc[i, "Title"] in ["Master", "Sir"]:     
        X_test.loc[i, "TitleNum"] = 1
    elif X_test.loc[i, "Title"] in ["Mrs", "Miss", "Ms"]:     
        X_test.loc[i, "TitleNum"] = -1
    elif X_test.loc[i, "Title"] in ["Lady"]:     
        X_test.loc[i, "TitleNum"] = -2
    else:
        X_test.loc[i, "TitleNum"] = 0

X_test = X_test[features]        
        
        


#X_test["Dependence"] = X_test["Parch"]*(X_test["Age"]-20)*(X_test["Age"]-60)

X_test["Sex"] = standardize_series(X_test["Sex"])
X_test["Age"] = standardize_series(X_test["Age"])
X_test["SibSp"] = standardize_series(X_test["SibSp"])
X_test["Parch"] = standardize_series(X_test["Parch"])
X_test["Fare"] = standardize_series(X_test["Fare"])
#X_test["Dependence"] = standardize_series(X_test["Dependence"])
X_test["TitleNum"] = standardize_series(X_test["TitleNum"])


#%%

"""

Fitting on ALL train data now, and predicting on the test

"""
print("Training for Kaggle Submission")

gamma = grid.best_params_['gamma']
C = grid.best_params_['C']    

titanicSVMgauss = svm.SVC(kernel = "rbf", C = C, gamma = gamma)

titanicSVMgauss.fit(X, labels) #fitting only on the training data 
print("Predicting for Kaggle Submission")
predictions = titanicSVMgauss.predict(X_test)
print(predictions)


#%% 

output = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": predictions})
output.to_csv('predictions.csv', index = False)