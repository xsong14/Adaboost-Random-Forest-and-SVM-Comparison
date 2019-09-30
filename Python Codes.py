

from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from collections import Counter
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn
import time
from sklearn import ensemble
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

#Datapreprocessing
data = pd.read_csv("bank-additional-full.csv",sep=';',na_values='unknown')
data = data.drop(['default'],axis=1)
data2=data.dropna()
data_y = data2.y
data_x = data2.drop(['y'],axis=1)
data_y[data_y == 'yes'] = 1
data_y[data_y == 'no'] = 0
data_x_dummy = pd.get_dummies(data_x)
min_max_scaler = preprocessing.StandardScaler().fit(data_x_dummy)
data_x_norm = min_max_scaler.transform(data_x_dummy)
c=Counter(data_y)
c
data_x_norm_df=pd.DataFrame(data_x_norm)
data_y_df=pd.DataFrame(data_y)
data_y_np=np.array(data_y_df)
data_y_df2=pd.DataFrame(data_y_np)


#Comparison between undersampling and raw data & Visualization by using PCA
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import tree
from imblearn.over_sampling import SMOTE
import math
import pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
data = pd.read_csv('bank-additional-full.csv', sep=';', na_values='unknown')
data2=data.drop(['default'],axis=1)
data3=data2.dropna(axis=0,how='any')
data_y=data3['y']
data['y'].value_counts().plot(kind="bar")
data_y[data_y=='yes']=1
data_y[data_y=='no']=0
data_x=data3.drop(['y'],axis=1)
x_dummy=pd.get_dummies(data_x)
x_dummy.shape
columnname=x_dummy.columns
scaler=preprocessing.StandardScaler()
scaler.fit(x_dummy)
x_minmax=scaler.transform(x_dummy)
x_minmax_scale=pd.DataFrame(x_minmax,columns=columnname)
data_y=np.array(data_y)
data_y=pd.DataFrame(data_y)
# undersampling decison tree
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i] and y_actual[i] ==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i] and y_actual[i] ==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1
    sensitivity=TP/(TP+FN)
    return sensitivity

training=[]
test=[]
lst=[20, 30, 50, 80, 100, 140, 180, 250 ]
sensitivity_train=[]
sensitivity_test=[]

for n in lst:
    score1=[]
    score2=[]
    lst_sensi_train=[]
    lst_sensi_test=[]
    for j in range(30):
        train_x, test_x, train_y, test_y = model_selection.train_test_split(x_minmax_scale,data_y, test_size=0.33)
    
        test_x=test_x.astype(float)
        test_y=test_y.astype(float)
        a= (train_y[train_y == 1]).dropna(axis=0, how='any')
        class1 = len(a)
        class0_indices = train_y[train_y == 0].index
        random_indices = np.random.choice(class0_indices,class1, replace=False)
        class1_indices = train_y[train_y == 1].index
        under_sample_indices = np.concatenate([class1_indices,random_indices])
        under_sample_y = data_y.loc[under_sample_indices]
        under_sample_x = x_minmax_scale.loc[under_sample_indices]
        
        clf = tree.DecisionTreeClassifier(criterion = "entropy", min_samples_split= n, min_samples_leaf=int(n/2),splitter="best")
        uder_sample_x=under_sample_x.astype(float)
        under_sample_y=under_sample_y.astype(float)
        clf.fit(under_sample_x, under_sample_y)
        
        predict_train=clf.predict(under_sample_x)
        predict_test=clf.predict(test_x)
        
        s1=clf.score(under_sample_x, under_sample_y)
        s2=clf.score(test_x,test_y)
        
        predict_train=list(predict_train)
        under_sample_y=np.array(under_sample_y)
        under_sample_y=list(under_sample_y)
        predict_test=list(predict_test)
        test_y=np.array(test_y)
        test_y=list(test_y)
        sensi_train= perf_measure(under_sample_y, predict_train)
        sensi_test=perf_measure(test_y, predict_test) 
        
        
        score1.append(s1)
        score2.append(s2)
        lst_sensi_train.append(sensi_train)
        lst_sensi_test.append(sensi_test)
    avg1= sum(score1)/30 
    avg2= sum(score2)/30
    sensi_train= sum(lst_sensi_train)/30
    sensi_test= sum(lst_sensi_test)/30
    training.append(avg1)
    test.append(avg2)
    sensitivity_train.append(sensi_train)
    sensitivity_test.append(sensi_test)

training
test
sensitivity_train
sensitivity_test


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x_minmax_scale)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, data_y], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [1, 0]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf[0] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

a= (data_y[data_y == 1]).dropna(axis=0, how='any')
class1 = len(a)
class0_indices = data_y[data_y == 0].index
random_indices = np.random.choice(class0_indices,class1, replace=False)
class1_indices = a.index
under_sample_indices = np.concatenate([class1_indices,random_indices])
under_sample_y = data_y.loc[under_sample_indices]
under_sample_x = x_minmax_scale.loc[under_sample_indices]

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(under_sample_x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

under_sample_y=np.array(under_sample_y)
under_sample_y=pd.DataFrame(under_sample_y)
finalDf = pd.concat([principalDf, under_sample_y], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [1, 0]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf[0] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

plt.plot(lst, training,'g--', label = 'training accuracy')
plt.plot(lst, test,'r--',label = 'test accuracy')
plt.ylabel('accuracy')
plt.xlabel('minimum number of cases in Np')
plt.legend()
plt.show()

training3=[]
test3=[]
lst3=[20, 30, 50, 80, 100, 140, 180, 250 ]
sensitivity_train3=[]
sensitivity_test3=[]

for n in lst:
    score1=[]
    score2=[]
    lst_sensi_train=[]
    lst_sensi_test=[]
    for j in range(30):
        train_x, test_x, train_y, test_y = model_selection.train_test_split(x_minmax_scale,data_y, test_size=0.33)
        
        clf = tree.DecisionTreeClassifier(criterion = "entropy", min_samples_split= n, min_samples_leaf=int(n/2),splitter="best")
        train_x=train_x.astype(float)
        train_y=train_y.astype(float)
        test_y=test_y.astype(float)
        clf.fit(train_x, train_y)
        
        predict_train=clf.predict(train_x)
        predict_test=clf.predict(test_x)
        
        s1=clf.score(train_x, train_y)
        s2=clf.score(test_x,test_y)
        
        predict_train=list(predict_train)
        under_sample_y=np.array(train_y)
        under_sample_y=list(train_y)
        predict_test=list(predict_test)
        test_y=np.array(test_y)
        test_y=list(test_y)
        train_y=np.array(train_y)
        train_y=list(train_y)
        sensi_train= perf_measure(train_y, predict_train)
        sensi_test=perf_measure(test_y, predict_test) 
        
        
        score1.append(s1)
        score2.append(s2)
        lst_sensi_train.append(sensi_train)
        lst_sensi_test.append(sensi_test)
    avg1= sum(score1)/30 
    avg2= sum(score2)/30
    sensi_train3= sum(lst_sensi_train)/30
    sensi_test3= sum(lst_sensi_test)/30
    training3.append(avg1)
    test3.append(avg2)
    sensitivity_train3.append(sensi_train3)
    sensitivity_test3.append(sensi_test3)

test3
training3
sensitivity_train3
sensitivity_test3


plt.plot(lst, training,'g--', label = 'balanced training')
plt.plot(lst, training3,'r--',label = 'unbalanced training')
plt.ylabel('accuracy')
plt.xlabel('minimum number of cases in Np')
plt.legend()
plt.show()

plt.plot(lst, test3,'g--', label = 'balanced testing')
plt.plot(lst, test,'r--',label = 'unbalanced testing')
plt.ylabel('accuracy')
plt.xlabel('minimum number of cases in Np')
plt.legend()
plt.show()

plt.plot(lst, sensitivity_train,'g--', label = 'balanced training')
plt.plot(lst, sensitivity_train3,'r--',label = 'unbalanced training')
plt.ylabel('sensitivity')
plt.xlabel('minimum number of cases in Np')
plt.legend()
plt.show()

plt.plot(lst, sensitivity_test,'g--', label = 'balanced testing')
plt.plot(lst, sensitivity_test3,'r--',label = 'unbalanced testing')
plt.ylabel('accuracy')
plt.xlabel('minimum number of cases in Np')
plt.legend()
plt.show()


#Apply Undersampling on the training set
score1=[]
score2=[]
for i in range(30):
    train_x, test_x, train_y, test_y = train_test_split(data_x_norm_df,data_y_df2, test_size=0.33)    
    
    test_x=test_x.astype(float)
    test_y=test_y.astype(float)
    
    a= (train_y[train_y == 1]).dropna(axis=0, how='any')
    class1 = len(a)
    class0_indices = train_y[train_y == 0].index
    random_indices = np.random.choice(class0_indices,class1, replace=False)
    class1_indices = train_y[train_y == 1].index
    under_sample_indices = np.concatenate([class1_indices,random_indices])
    
    under_sample_y = data_y_df2.loc[under_sample_indices]
    under_sample_x = data_x_norm_df.loc[under_sample_indices]
    
    clf = tree.DecisionTreeClassifier(criterion = "entropy", min_samples_split=190, min_samples_leaf=95,splitter="best")
    under_sample_x=under_sample_x.astype(float)
    under_sample_y=under_sample_y.astype(float)
    clf.fit(under_sample_x, under_sample_y)
    
    s1=clf.score(under_sample_x, under_sample_y)
    s2=clf.score(test_x,test_y)
    score1.append(s1)
    score2.append(s2)


#Testing linear and non-linear SVM
def sensitivity(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i] and y_actual[i] ==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i] and y_actual[i] ==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1
    sensi=TP/(TP+FN)
return sensi
C_2d_range_lsvm = [1e-3,1e-2,0.1,1,10]

Accurate_lsvm_train=[]
Accurate_lsvm_test=[]
Sensitivity_lsvm_train=[]
Sensitivity_lsvm_test=[]
lsvm_time=[]

for C in C_2d_range_lsvm:
    model2 = SVC(kernel='linear', C=C)
    start_time_train_lsvm = time.time()
    model2.fit(under_sample_x, under_sample_y)
    t_train_lsvm=time.time()-start_time_train_lsvm
    lsvm_time.append(t_train_lsvm)
    Accurate_lsvm_train.append(model2.score(under_sample_x, under_sample_y))
    Accurate_lsvm_test.append(model2.score(test_x, test_y))
        
    pred_train_lsvm=model2.predict(under_sample_x)
    data_y_train_np=np.array(under_sample_y)
    Sensitivity_lsvm_train.append(sensitivity(data_y_train_np,pred_train_lsvm))
    pred_test_lsvm=model2.predict(test_x)
    data_y_test_np=np.array(test_y)
    Sensitivity_lsvm_test.append(sensitivity(data_y_test_np,pred_test_lsvm))


print (Accurate_lsvm_train)
print (Accurate_lsvm_test)
print (Sensitivity_lsvm_train)
print (Sensitivity_lsvm_test)
print (lsvm_time)

C_2d_range = [1e-3,1e-2,0.1,1,10,100,1e3]
gamma_2d_range = [1e-3,1e-2,0.1,1,10,100,1e3]
Accuracy_train = []
Accuracy_test = []
sensitivity_train = []
sensitivity_test = []
time_train = []

for C in C_2d_range:
    for gamma in gamma_2d_range:
        model = SVC(C=C, gamma=gamma)
        start_time_train = time.time()
        model.fit(under_sample_x, under_sample_y)
        t_train=time.time()-start_time_train
        time_train.append(t_train)
        Accuracy_train.append(model.score(under_sample_x, under_sample_y))
        Accuracy_test.append(model.score(test_x, test_y))
        
        pred_train=model.predict(under_sample_x)
        data_y_train_np=np.array(under_sample_y)
        sensitivity_train.append(sensitivity(data_y_train_np,pred_train))
        pred_test=model.predict(test_x)
        data_y_test_np=np.array(test_y)
        sensitivity_test.append(sensitivity(data_y_test_np,pred_test))

Accuracy_train=np.array(Accuracy_train)
Accuracy_train=np.reshape(Accuracy_train, (7, 7))
Accuracy_test=np.array(Accuracy_test)
Accuracy_test=np.reshape(Accuracy_test, (7, 7))
sensitivity_train=np.array(sensitivity_train)
sensitivity_train=np.reshape(sensitivity_train, (7, 7))
sensitivity_test=np.array(sensitivity_test)
sensitivity_test=np.reshape(sensitivity_test, (7, 7))
time_train=np.array(time_train)
time_train=np.reshape(time_train, (7, 7))

print (Accuracy_train)
print (Accuracy_test)
print (sensitivity_train)
print (sensitivity_test)
print (time_train)

seaborn.heatmap(Accuracy_train)
seaborn.heatmap(Accuracy_test)
seaborn.heatmap(sensitivity_train)
seaborn.heatmap(sensitivity_test)

#Testing Adaboost
clf_A=ensemble.AdaBoostClassifier()
Accuracy_train_A=[]
Accuracy_test_A=[]
n=[]
for i in range(1,200):
    clf_A=ensemble.AdaBoostClassifier(n_estimators=i)
    clf_A.fit(under_sample_x, under_sample_y)
    A=clf_A.score(under_sample_x, under_sample_y)
    
    Accuracy_train_i=A
    Accuracy_train_A.append(Accuracy_train_i)    
    n.append(i)
    
    S=clf_A.score(test_x,test_y)
    
    Accuracy_test_i=S
Accuracy_test_A.append(Accuracy_test_i)


plt.plot(n, Accuracy_train_A,'g--', label = 'training Accuracy')
plt.plot(n, Accuracy_test_A,'r--',label = 'test Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('# of interation')
plt.legend()
plt.show()


def sensitivity(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i] and y_actual[i] ==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i] and y_actual[i] ==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1
    sensi=TP/(TP+FN)
return sensi

clf_A=ensemble.AdaBoostClassifier(n_estimators=162)
start_time_train_ada = time.time()
clf_A.fit(under_sample_x, under_sample_y)
ada_time=time.time()-start_time_train_ada

Accurate_ada_train=(clf_A.score(under_sample_x, under_sample_y))
Accurate_ada_test=(clf_A.score(test_x, test_y))
        
pred_train_ada=clf_A.predict(under_sample_x)
data_y_train_ada=np.array(under_sample_y)
Sensitivity_ada_train=(sensitivity(data_y_train_ada,pred_train_ada))
pred_test_ada=clf_A.predict(test_x)
data_y_test_ada=np.array(test_y)
Sensitivity_ada_test=(sensitivity(data_y_test_ada,pred_test_ada))


print (Accurate_ada_train)
print (Accurate_ada_test)
print (Sensitivity_ada_train)
print (Sensitivity_ada_test)
print (ada_time)

#Testing Naive Bayes
score1_nb=[]
score2_nb=[]
sensitivity_nb_train=[]
sensitivity_nb_test=[]
nb_time=[]
n_nb=[]
for i in range(1,101,2):
    train_x, test_x, train_y, test_y = train_test_split(data_x_norm_df,data_y_df2, test_size=0.33)    
    
    test_x=test_x.astype(float)
    test_y=test_y.astype(float)
    
    a= (train_y[train_y == 1]).dropna(axis=0, how='any')
    class1 = len(a)
    class0_indices = train_y[train_y == 0].index
    random_indices = np.random.choice(class0_indices,class1, replace=False)
    class1_indices = train_y[train_y == 1].index
    under_sample_indices = np.concatenate([class1_indices,random_indices]) 
    under_sample_y = data_y_df2.loc[under_sample_indices]
    under_sample_x = data_x_norm_df.loc[under_sample_indices]
    under_sample_x=under_sample_x.astype(float)
    under_sample_y=under_sample_y.astype(float)    
    n_nb.append(i)
    gnb = GaussianNB()
    start_time_train_nb = time.time()
    gnb.fit(under_sample_x, under_sample_y)     
    t_train_nb=time.time()-start_time_train_nb
    nb_time.append(t_train_nb)    
    score1_nb.append(gnb.score(under_sample_x, under_sample_y))
    score2_nb.append(gnb.score(test_x,test_y))
    
    pred_train_nb=gnb.predict(under_sample_x)
    data_y_train_nb=np.array(under_sample_y)
    sensitivity_nb_train.append(sensitivity(data_y_train_nb,pred_train_nb))
    pred_test_nb=gnb.predict(test_x)
    data_y_test_nb=np.array(test_y)
    sensitivity_nb_test.append(sensitivity(data_y_test_nb,pred_test_nb))
    
Accurate_nb_train=sum(score1_nb)/len(score1_nb)
Accurate_nb_test=sum(score2_nb)/len(score2_nb)
Sensitivity_nb_train=sum(sensitivity_nb_train)/len(sensitivity_nb_train)
Sensitivity_nb_test=sum(sensitivity_nb_test)/len(sensitivity_nb_test)


print('MeanAccuracyTrain: {:7.4f}. CI: [{:7.4f}, {:7.4f}]'.format(Accurate_nb_train, mean_train-1.96*var_train/math.sqrt(30),mean_train+1.96*var_train/math.sqrt(30)))
print('MeanAccuracyTest: {:7.4f}. CI: [{:7.4f}, {:7.4f}]'.format(Accurate_nb_test, mean_test-1.96*var_test/math.sqrt(30),mean_test+1.96*var_train/math.sqrt(30)))
print (Sensitivity_nb_train)
print (Sensitivity_nb_test)
print (sum(nb_time))

plt.plot(n_nb, sensitivity_nb_train,'g--', label = 'train sensitivity')
plt.plot(n_nb, sensitivity_nb_test,'r--',label = 'test sensitivity')
plt.ylabel('sensitivity')
plt.xlabel('# of iterations')
plt.legend()
plt.show()


#Random Forest
Accurate_rf_train=[]
Accurate_rf_test=[]
sensitivity_rf_train=[]
sensitivity_rf_test=[]
rf_time=[]
N=[]
for n2 in range(50,92,2):
    train_x, test_x, train_y, test_y = train_test_split(data_x_norm_df,data_y_df2, test_size=0.33)    
    
    test_x=test_x.astype(float)
    test_y=test_y.astype(float)
    
    a= (train_y[train_y == 1]).dropna(axis=0, how='any')
    class1 = len(a)
    class0_indices = train_y[train_y == 0].index
    random_indices = np.random.choice(class0_indices,class1, replace=False)
    class1_indices = train_y[train_y == 1].index
    under_sample_indices = np.concatenate([class1_indices,random_indices]) 
    under_sample_y = data_y_df2.loc[under_sample_indices]
    under_sample_x = data_x_norm_df.loc[under_sample_indices]
    under_sample_x=under_sample_x.astype(float)
    under_sample_y=under_sample_y.astype(float)    
    
    RF=RandomForestClassifier(n_estimators=173, random_state = 20,min_samples_split=n2,min_samples_leaf=n2//2)
    
    start_time_train_rf = time.time()
    RF.fit(under_sample_x, under_sample_y)     
    t_train_rf=time.time()-start_time_train_rf
    rf_time.append(t_train_rf)    
    Accurate_rf_train.append(1-RF.score(under_sample_x, under_sample_y))
    Accurate_rf_test.append(1-RF.score(test_x,test_y))
    N.append(n2)
    pred_train_rf=RF.predict(under_sample_x)
    data_y_train_rf=np.array(under_sample_y)
    sensitivity_rf_train.append(sensitivity(data_y_train_rf,pred_train_rf))
    pred_test_rf=RF.predict(test_x)
    data_y_test_rf=np.array(test_y)
sensitivity_rf_test.append(sensitivity(data_y_test_rf,pred_test_rf))

plt.plot(N, Accurate_rf_train,'g--', label = 'train error')
plt.plot(N, Accurate_rf_test,'r--',label = 'test error')
plt.ylabel('Error')
plt.xlabel('value of min_samples_split')
plt.legend()
plt.show()

Accurate_rf_train=[]
Accurate_rf_test=[]
sensitivity_rf_train=[]
sensitivity_rf_test=[]
rf_time=[]
N=[]
for n2 in range(100,201,5):
    train_x, test_x, train_y, test_y = train_test_split(data_x_norm_df,data_y_df2, test_size=0.33)    
    
    test_x=test_x.astype(float)
    test_y=test_y.astype(float)
    
    a= (train_y[train_y == 1]).dropna(axis=0, how='any')
    class1 = len(a)
    class0_indices = train_y[train_y == 0].index
    random_indices = np.random.choice(class0_indices,class1, replace=False)
    class1_indices = train_y[train_y == 1].index
    under_sample_indices = np.concatenate([class1_indices,random_indices]) 
    under_sample_y = data_y_df2.loc[under_sample_indices]
    under_sample_x = data_x_norm_df.loc[under_sample_indices]
    under_sample_x=under_sample_x.astype(float)
    under_sample_y=under_sample_y.astype(float)    
    
    RF=RandomForestClassifier(n_estimators=100, random_state = 20,min_samples_split=n2,min_samples_leaf=n2//2,max_depth=5)
    
    start_time_train_rf = time.time()
    RF.fit(under_sample_x, under_sample_y)     
    t_train_rf=time.time()-start_time_train_rf
    rf_time.append(t_train_rf)    
    Accurate_rf_train.append(1-RF.score(under_sample_x, under_sample_y))
    Accurate_rf_test.append(1-RF.score(test_x,test_y))
    N.append(n2)
    pred_train_rf=RF.predict(under_sample_x)
    data_y_train_rf=np.array(under_sample_y)
    sensitivity_rf_train.append(sensitivity(data_y_train_rf,pred_train_rf))
    pred_test_rf=RF.predict(test_x)
    data_y_test_rf=np.array(test_y)
    sensitivity_rf_test.append(sensitivity(data_y_test_rf,pred_test_rf))

plt.plot(N, Accurate_rf_train,'g--', label = 'training error')
plt.plot(N, Accurate_rf_test,'r--',label = 'test error')
plt.ylabel('Error')
plt.xlabel('# of min_samples_split')
plt.legend()
plt.show()



#Run all the result

# coding: utf-8

# In[2]:


from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from collections import Counter
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import seaborn
import time
from sklearn import ensemble
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.metrics import roc_auc_score


# # Data preprocessing

# In[3]:


data = pd.read_csv("bank-additional-full.csv",sep=';',na_values='unknown')
data = data.drop(['default'],axis=1)
data2=data.dropna()
data_y = data2.y
data_x = data2.drop(['y'],axis=1)
data_y[data_y == 'yes'] = 1
data_y[data_y == 'no'] = 0


# In[4]:


data_x_dummy = pd.get_dummies(data_x)
min_max_scaler = preprocessing.StandardScaler().fit(data_x_dummy)
data_x_norm = min_max_scaler.transform(data_x_dummy)


# # Undersampling

# In[5]:


data_x_norm_df=pd.DataFrame(data_x_norm)
data_y_df=pd.DataFrame(data_y)
data_y_np=np.array(data_y_df)
data_y_df2=pd.DataFrame(data_y_np)


# In[6]:


score1=[]
score2=[]
for i in range(30):
    train_x, test_x, train_y, test_y = train_test_split(data_x_norm_df,data_y_df2, test_size=0.33)    
    
    test_x=test_x.astype(float)
    test_y=test_y.astype(float)
    
    a= (train_y[train_y == 1]).dropna(axis=0, how='any')
    class1 = len(a)
    class0_indices = train_y[train_y == 0].index
    random_indices = np.random.choice(class0_indices,class1, replace=False)
    class1_indices = train_y[train_y == 1].index
    under_sample_indices = np.concatenate([class1_indices,random_indices])
    
    under_sample_y = data_y_df2.loc[under_sample_indices]
    under_sample_x = data_x_norm_df.loc[under_sample_indices]
    
    clf = tree.DecisionTreeClassifier(criterion = "entropy", min_samples_split=190, min_samples_leaf=95,splitter="best")
    under_sample_x=under_sample_x.astype(float)
    under_sample_y=under_sample_y.astype(float)
    clf.fit(under_sample_x, under_sample_y)
    
    s1=clf.score(under_sample_x, under_sample_y)
    s2=clf.score(test_x,test_y)
    score1.append(s1)
    score2.append(s2)


# In[7]:


def sensitivity(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i] and y_actual[i] ==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i] and y_actual[i] ==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1
    sensi=TP/(TP+FN)
    return sensi


# # SVM

# In[8]:


C_2d_range = [1]
gamma_2d_range = [0.1]


# In[9]:


Accuracy_train = []
Accuracy_test = []
sensitivity_train = []
sensitivity_test = []
time_train = []

for C in C_2d_range:
    for gamma in gamma_2d_range:
        model = SVC(C=C, gamma=gamma,probability=True)
        start_time_train = time.time()
        model.fit(under_sample_x, under_sample_y)
        t_train=time.time()-start_time_train
        time_train.append(t_train)
        Accuracy_train.append(model.score(under_sample_x, under_sample_y))
        Accuracy_test.append(model.score(test_x, test_y))
        
        pred_train=model.predict(under_sample_x)
        data_y_train_np=np.array(under_sample_y)
        sensitivity_train.append(sensitivity(data_y_train_np,pred_train))
        pred_test=model.predict(test_x)
        data_y_test_np=np.array(test_y)
        sensitivity_test.append(sensitivity(data_y_test_np,pred_test))

print (Accuracy_train)
print (Accuracy_test)
print (sensitivity_train)
print (sensitivity_test)
print (time_train)


# # Adaboost

# In[10]:


clf_A=ensemble.AdaBoostClassifier(n_estimators=162)
start_time_train_ada = time.time()
clf_A.fit(under_sample_x, under_sample_y)
ada_time=time.time()-start_time_train_ada

Accurate_ada_train=(clf_A.score(under_sample_x, under_sample_y))
Accurate_ada_test=(clf_A.score(test_x, test_y))
        
pred_train_ada=clf_A.predict(under_sample_x)
data_y_train_ada=np.array(under_sample_y)
Sensitivity_ada_train=(sensitivity(data_y_train_ada,pred_train_ada))
pred_test_ada=clf_A.predict(test_x)
data_y_test_ada=np.array(test_y)
Sensitivity_ada_test=(sensitivity(data_y_test_ada,pred_test_ada))


print (Accurate_ada_train)
print (Accurate_ada_test)
print (Sensitivity_ada_train)
print (Sensitivity_ada_test)
print (ada_time)


# # Naive Bayes

# In[11]:


score1_nb=[]
score2_nb=[]
sensitivity_nb_train=[]
sensitivity_nb_test=[]
nb_time=[]

for i in range(83):
    train_x, test_x, train_y, test_y = train_test_split(data_x_norm_df,data_y_df2, test_size=0.33)    
    
    test_x=test_x.astype(float)
    test_y=test_y.astype(float)
    
    a= (train_y[train_y == 1]).dropna(axis=0, how='any')
    class1 = len(a)
    class0_indices = train_y[train_y == 0].index
    random_indices = np.random.choice(class0_indices,class1, replace=False)
    class1_indices = train_y[train_y == 1].index
    under_sample_indices = np.concatenate([class1_indices,random_indices]) 
    under_sample_y = data_y_df2.loc[under_sample_indices]
    under_sample_x = data_x_norm_df.loc[under_sample_indices]
    under_sample_x=under_sample_x.astype(float)
    under_sample_y=under_sample_y.astype(float)    
    
    gnb = GaussianNB()
    start_time_train_nb = time.time()
    gnb.fit(under_sample_x, under_sample_y)     
    t_train_nb=time.time()-start_time_train_nb
    nb_time.append(t_train_nb)    
    score1_nb.append(gnb.score(under_sample_x, under_sample_y))
    score2_nb.append(gnb.score(test_x,test_y))
    
    pred_train_nb=gnb.predict(under_sample_x)
    data_y_train_nb=np.array(under_sample_y)
    sensitivity_nb_train.append(sensitivity(data_y_train_nb,pred_train_nb))
    pred_test_nb=gnb.predict(test_x)
    data_y_test_nb=np.array(test_y)
    sensitivity_nb_test.append(sensitivity(data_y_test_nb,pred_test_nb))
    
Accurate_nb_train=sum(score1_nb)/len(score1_nb)
Accurate_nb_test=sum(score2_nb)/len(score2_nb)
Sensitivity_nb_train=sum(sensitivity_nb_train)/len(sensitivity_nb_train)
Sensitivity_nb_test=sum(sensitivity_nb_test)/len(sensitivity_nb_test)
var_train=np.var(score1_nb)
var_test=np.var(score2_nb)


print('MeanAccuracyTrain: {:7.4f}. CI: [{:7.4f}, {:7.4f}]'.format(Accurate_nb_train, Accurate_nb_train-1.96*var_train/math.sqrt(30),Accurate_nb_train+1.96*var_train/math.sqrt(30)))
print('MeanAccuracyTest: {:7.4f}. CI: [{:7.4f}, {:7.4f}]'.format(Accurate_nb_test, Accurate_nb_test-1.96*var_test/math.sqrt(30),Accurate_nb_test+1.96*var_train/math.sqrt(30)))
print (Sensitivity_nb_train)
print (Sensitivity_nb_test)
print (sum(nb_time))


# # Random Forest

# In[26]:


Accurate_rf_train=[]
Accurate_rf_test=[]
sensitivity_rf_train=[]
sensitivity_rf_test=[]
rf_time=[]
    
RF=RandomForestClassifier(n_estimators=173, random_state = 20,min_samples_split=53,min_samples_leaf=26,max_depth=5)
    
start_time_train_rf = time.time()
RF.fit(under_sample_x, under_sample_y)     
t_train_rf=time.time()-start_time_train_rf
rf_time.append(t_train_rf)    
Accurate_rf_train.append(RF.score(under_sample_x, under_sample_y))
Accurate_rf_test.append(RF.score(test_x,test_y))
pred_train_rf=RF.predict(under_sample_x)
data_y_train_rf=np.array(under_sample_y)
sensitivity_rf_train.append(sensitivity(data_y_train_rf,pred_train_rf))
pred_test_rf=RF.predict(test_x)
data_y_test_rf=np.array(test_y)
sensitivity_rf_test.append(sensitivity(data_y_test_rf,pred_test_rf))

print (Accurate_rf_train)
print (Accurate_rf_test)
print (sensitivity_rf_train)
print (sensitivity_rf_test)
print (rf_time)


# # ROC

# In[13]:


score1 = model.fit(under_sample_x, under_sample_y).predict_proba(under_sample_x)


# In[14]:


score2 = clf_A.fit(under_sample_x, under_sample_y).predict_proba(under_sample_x)
score3 = gnb.fit(under_sample_x, under_sample_y).predict_proba(under_sample_x)
score4 = RF.fit(under_sample_x, under_sample_y).predict_proba(under_sample_x)


# In[15]:


fpr_svm, tpr_svm, thresholds_svm = metrics.roc_curve(under_sample_y, score1[:,1])
fpr_ada, tpr_ada, thresholds_ada = metrics.roc_curve(under_sample_y, score2[:,1])
fpr_nb, tpr_nb, thresholds_nb = metrics.roc_curve(under_sample_y, score3[:,1])
fpr_rf, tpr_rf, thresholds_rf = metrics.roc_curve(under_sample_y, score4[:,1])


# In[16]:


roc_auc_svm = auc(fpr_svm, tpr_svm)
roc_auc_ada = auc(fpr_ada, tpr_ada)
roc_auc_nb = auc(fpr_nb, tpr_nb)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure()
plt.plot(fpr_svm, tpr_svm, color='darkorange',  label='Nonlinear-SVM (area = {0:0.2f})'''.format(roc_auc_svm))
plt.plot(fpr_ada, tpr_ada, color='maroon', label='Adaboost (area = {0:0.2f})'''.format(roc_auc_ada))
plt.plot(fpr_nb, tpr_nb, color='olive', label='Naive Bayes (area = {0:0.2f})'''.format(roc_auc_nb))
plt.plot(fpr_rf, tpr_rf, color='darkblue', label='RandomForest (area = {0:0.2f})'''.format(roc_auc_rf))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[21]:


sum(thresholds_svm)/len(thresholds_svm)


# In[22]:


sum(thresholds_ada)/len(thresholds_ada)


# In[23]:


sum(thresholds_nb)/len(thresholds_nb)


# In[24]:


sum(thresholds_rf)/len(thresholds_rf)



