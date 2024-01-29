#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:
from functools import partial
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from os import listdir
from os.path import isfile, join
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble  import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import pandas as pd
from pandas import Series, DataFrame
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from scipy.io.arff import loadarff
from scipy import interp
import json, logging, tempfile, sys, codecs, math, io, os,zipfile, arff, time, copy, csv,pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import roc_curve, auc
from hyperopt import fmin, tpe, hp, rand, STATUS_OK, Trials
from imblearn.over_sampling import (SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, SMOTENC,SMOTEN,
                                    KMeansSMOTE, RandomOverSampler)
from imblearn.under_sampling import (ClusterCentroids, RandomUnderSampler,
                                     NearMiss, TomekLinks,
                                     InstanceHardnessThreshold,
                                     CondensedNearestNeighbour,
                                     EditedNearestNeighbours,
                                     RepeatedEditedNearestNeighbours,
                                     AllKNN,
                                     NeighbourhoodCleaningRule,
                                     OneSidedSelection)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.metrics import geometric_mean_score
from scipy.spatial import distance
from sklearn import metrics
import glob
# In[2]:
dataset1= str(sys.argv[1])
seed= int(sys.argv[2])
acctxt= str(sys.argv[3])
Mclasstxt= str(sys.argv[4])
dsfold= int(sys.argv[5])
HPOalg = 'TPE'
DataFolder='./DATA'
HomeFolder='./RESULTS'

dataName=dataset1#'Top' if dataset=='top' else 'Bot'
dataset ='Top_Exp.csv' if dataset1=='Top' else 'Bot_Exp.csv'

trainIDs,testIDs='',''
for file in glob.glob('./DATA/TrainTest/'+dataName+'-exp'+str(dsfold)+'/'+'*_ASIS_'+dataName+'_exp'+str(dsfold)+'_learn.txt'):
    #print(file)
    trainIDs=file
for file in glob.glob('./DATA/TrainTest/'+dataName+'-exp'+str(dsfold)+'/'+'*_ASIS_'+dataName+'_exp'+str(dsfold)+'_test.txt'):
    #print(file)
    testIDs=file
file= DataFolder+'/'+dataset
def accuracy_report(y_test_ex, y_hat_ex):
    accuracy=metrics.accuracy_score(y_test_ex, y_hat_ex)
    f1_macro=metrics.f1_score(y_test_ex, y_hat_ex, average='macro')
    f1_micro=metrics.f1_score(y_test_ex, y_hat_ex, average='micro')
    f1_weighted=metrics.f1_score(y_test_ex, y_hat_ex, average='weighted')
    gm_macro = geometric_mean_score(y_test_ex, y_hat_ex, average='macro')
    gm_micro = geometric_mean_score(y_test_ex, y_hat_ex, average='micro')
    gm_weighted = geometric_mean_score(y_test_ex, y_hat_ex, average='weighted')
    
    _re={'acc':accuracy,'F1_weight':f1_weighted,'F1_micro':f1_micro,
         'F1_macro':f1_macro,'GM_weight':gm_weighted,'GM_micro':gm_micro,
         'GM_macro':gm_macro}
    return _re
def cost_matrix(y_test, y_hat):
    _before =accuracy_report(y_test,y_hat)
    global labelname
    matrix_names = ['10', '20', '40', '60', '70', '90', '100', '110', '130', '140', '160', '180', '190', '200', '220', '230',
            '250', '260', '280', '290', '380', '390', '400', '410', '420', '430', '460']
    cost_matrix = pd.DataFrame(np.ones((27, 27)), columns = matrix_names, index = matrix_names)
    cost_matrix['130'] = [5, 8, 5, 8, 5, 5, 5, 5, 0, 5, 10, 10, 5, 5, 10, 8, 10, 10, 10, 5, 8, 5, 10, 10, 10, 10, 10]
    cost_matrix['140'] = [5, 8, 5, 8, 5, 5, 5, 5, 5, 0, 10, 10, 5, 5, 10, 8, 10, 10, 10, 5, 8, 5, 10, 10, 10, 10, 10]
    cost_matrix.iloc[8, :] = [5, 10, 5, 8, 5, 5, 5, 5, 0, 5, 8, 8, 5, 5, 8, 10, 10, 8, 10, 5, 10, 8, 10, 10, 10, 10, 10]
    cost_matrix.iloc[9, :] = [5, 10, 5, 8, 5, 5, 5, 5, 5, 0, 8, 8, 5, 5, 8, 10, 10, 8, 10, 5, 10, 8, 10, 10, 10, 10, 10]
    for i in list(range(27)):
        cost_matrix.iloc[i, i] = 0

    name_string = list(map(str, labelname))
    submatrix = cost_matrix[name_string]
    sub_costmatrix = submatrix[submatrix.index.isin(name_string)]
    confusion_matrix = metrics.confusion_matrix(y_test, y_hat, labels = labelname)
    confusion_label = pd.DataFrame(confusion_matrix, index=labelname, columns=labelname)
    
    for x in labelname:
        for z in labelname:
            _scount=confusion_label.loc[x][z]
            if _scount!=0:
                _w= cost_matrix.loc[str(x)][str(z)]
                if (_w!=0 and _w!=1):  
                    #print(x,z,_scount,_w)
                    #add more samples
                    _w=_w-1
                    y_test=y_test.append(pd.Series([x]*int((_scount*_w))))
                    y_hat=y_hat.append(pd.Series([z]*int((_scount*_w))))
    _after=accuracy_report(y_test,y_hat)
    return _before,_after
def getSP(seed):
    ##====SVM =====
    HPOspace = hp.choice('classifier_type', [
        {	'random_state': seed,
            'classifier': hp.choice('classifier',[
                {
                    'name': 'SVM',
                    'probability': hp.choice("probability", [True, False]),                
                    'C': hp.uniform('C', 0.03125 , 200 ),
                    'kernel': hp.choice('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),                
                    "degree": hp.randint("degree", 2,5),
                    "gamma": hp.choice('gamma',['auto','value','scale']),
                    'gamma_value': hp.uniform('gamma_value', 3.1E-05, 8),
                    "coef0": hp.uniform('coef0', -1, 1),
                    "shrinking": hp.choice("shrinking", [True, False]),
                    "tol": hp.uniform('tol_svm', 1e-05, 1e-01)#NEW
                },
                {
                    'name': 'RF',
                    'n_estimators': hp.randint("n_estimators", 1, 150),
                    'criterion': hp.choice('criterion', ["gini", "entropy"]),
                    'max_features': hp.choice('max_features_RF', [1, 'sqrt','log2',None]),               
                    'min_samples_split': hp.randint('min_samples_split', 2, 20),
                    'min_samples_leaf': hp.randint('min_samples_leaf', 1, 20),
                    'bootstrap': hp.choice('bootstrap',[True, False]),
                    'class_weight':hp.choice('class_weight',['balanced','balanced_subsample',None]),
                },
                {
                    'name': 'KNN',
                    'n_neighbors': hp.randint("n_neighbors_knn", 1, 51),
                    'weights': hp.choice('weights', ["uniform", "distance"]),
                    'algorithm': hp.choice('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                    'p': hp.randint("p_value", 0, 20),
                },
                {
                    'name': 'DTC',
                    'criterion': hp.choice("criterion_dtc", ["gini", "entropy"]),
                    'max_features': hp.choice('max_features_dtc', [1, 'sqrt','log2',None]),
                    'max_depth': hp.choice('max_depth_dtc', range(2,20)),
                    'min_samples_split': hp.randint('min_samples_split_dtc', 2,20),
                    'min_samples_leaf': hp.randint('min_samples_leaf_dtc',1,20)     
                },
                {
                    'name': 'LR',
                    'C': hp.uniform('C_lr', 0.03125 , 100 ),
                    'penalty_solver': hp.choice("penalty_lr", ["l1+liblinear","l1+saga","l2+newton-cg","l2+lbfgs",
                                                               "l2+liblinear","l2+sag","l2+saga","elasticnet+saga",
                                                               "none+newton-cg","none+lbfgs","none+sag","none+saga"]),
                    'tol': hp.uniform('tol_lr', 1e-05, 1e-01),
                    'l1_ratio': hp.uniform('l1_ratio', 1e-09, 1)
                }]),

            'sub' : hp.choice('resampling_type',[
                {
                    'type':'NO'
                },
                #Over sampling
                {
                    'type': 'SMOTE',
                    'k_neighbors': hp.randint('k_neighbors_SMOTE',1,10),
                    'sampling_strategy':hp.choice('sampling_strategy1',['minority','not minority','not majority','all','auto'])
                },
                {
                    'type': 'BorderlineSMOTE',
                    'k_neighbors': hp.randint('k_neighbors_Borderline',1,10),
                    'm_neighbors': hp.randint('m_neighbors_Borderline',1,10),
                    'kind' :  hp.choice('kind', ['borderline-1', 'borderline-2']),
                    'sampling_strategy':hp.choice('sampling_strategy2',['minority','not minority','not majority','all','auto'])
                },
                {
                    'type': 'SMOTEN',
                    'sampling_strategy':hp.choice('sampling_strategy3',['minority','not minority','not majority','all','auto']),
                    'k_neighbors': hp.randint('k_neighbors_SMOTEN',1,10), 
                },
                {
                    'type': 'SMOTENC',
                    'categorical_features': True,
                    'k_neighbors': hp.randint('k_neighbors_SMOTENC',1,10), 
                    'sampling_strategy':hp.choice('sampling_strategy4',['minority','not minority','not majority','all','auto'])
                },
                {
                    'type': 'SVMSMOTE',  
                    'k_neighbors': hp.randint('k_neighbors_SVMSMOTE',1,10), 
                    'm_neighbors': hp.randint('m_neighbors_SVMSMOTE',1,10),                
                    'out_step': hp.uniform('out_step', 0, 1),
                    'sampling_strategy':hp.choice('sampling_strategy5',['minority','not minority','not majority','all','auto'])
                },
                {
                    'type': 'KMeansSMOTE',  
                    'k_neighbors': hp.randint('k_neighbors_KMeansSMOTE',1,10), 
                    'cluster_balance_threshold': hp.uniform('cluster_balance_threshold', 1e-2, 1), 
                    'sampling_strategy':hp.choice('sampling_strategy6',['minority','not minority','not majority','all','auto'])
                },
                {
                    'type': 'ADASYN',
                    'n_neighbors' : hp.randint('n_neighbors_ADASYN',1,10),
                    'sampling_strategy':hp.choice('sampling_strategy7',['minority','not minority','not majority','all','auto'])
                },    
                {
                    'type': 'RandomOverSampler',
                    'sampling_strategy':hp.choice('sampling_strategy8',['minority','not minority','not majority','all','auto'])
                },
                #COMBINE RESAMPLING
                {
                    'type': 'SMOTEENN',
                    'sampling_strategy':hp.choice('sampling_strategy9',['minority','not minority','not majority','all','auto'])
                },
                {
                    'type': 'SMOTETomek',
                    'sampling_strategy':hp.choice('sampling_strategy10',['minority','not minority','not majority','all','auto'])
                },   
                 #UNDER RESAMPLING
                {
                    'type': 'CondensedNearestNeighbour',
                    'n_neighbors' : hp.randint('n_neighbors_CNN',1,50),
                    'n_seeds_S' : hp.randint('n_seeds_S_CNN',1,50),
                    'sampling_strategy':hp.choice('sampling_strategy11',['majority','not minority','not majority','all','auto'])
                },
                {
                    'type': 'EditedNearestNeighbours',
                    'n_neighbors' : hp.randint('n_neighbors_ENN',1,20),
                    'kind_sel' : hp.choice('kind_sel_ENN',['all','mode']),
                    'sampling_strategy':hp.choice('sampling_strategy12',['majority','not minority','not majority','all','auto'])
                },
                {
                    'type': 'RepeatedEditedNearestNeighbours',
                    'n_neighbors' : hp.randint('n_neighbors_RNN',1,20),
                    'kind_sel' : hp.choice('kind_sel_RNN',['all','mode']),
                    'sampling_strategy':hp.choice('sampling_strategy13',['majority','not minority','not majority','all','auto'])
                },
                {
                    'type': 'AllKNN',
                    'n_neighbors' : hp.randint('n_neighbors_AKNN',1,20),
                    'kind_sel' : hp.choice('kind_sel_AKNN',['all','mode']),
                    'allow_minority' : hp.choice('allow_minority_AKNN', [True, False]),
                    'sampling_strategy':hp.choice('sampling_strategy14',['majority','not minority','not majority','all','auto'])
                },
                {
                    'type': 'InstanceHardnessThreshold',
                    'estimator': hp.choice('estimator_IHTh', ['knn', 'decision-tree', 'adaboost','gradient-boosting','linear-svm', None]),
                    'cv' : hp.randint('cv_IHTh',2,10,),
                    'sampling_strategy':hp.choice('sampling_strategy15',['majority','not minority','not majority','all','auto'])
                },
                {
                    'type': 'NearMiss',
                    'version' : hp.choice('version_NM',[1,2,3]),
                    'n_neighbors' : hp.randint('n_neighbors_NM',1,20),
                    'n_neighbors_ver3' : hp.randint('n_neighbors_ver3_NM',1,20),
                    'sampling_strategy':hp.choice('sampling_strategy16',['majority','not minority','not majority','all','auto'])
                },
                {
                    'type': 'NeighbourhoodCleaningRule',
                    'n_neighbors' : hp.randint('n_neighbors_NCR',1,20),
                    'threshold_cleaning' : hp.uniform('threshold_cleaning_NCR',0,1),
                    'sampling_strategy':hp.choice('sampling_strategy17',['majority','not minority','not majority','all','auto'])
                },
                {
                    'type': 'OneSidedSelection',
                    'n_neighbors' : hp.randint('n_neighbors_OSS',1,20),
                    'n_seeds_S' : hp.randint('n_seeds_S_OSS',1,20),
                    'sampling_strategy':hp.choice('sampling_strategy18',['majority','not minority','not majority','all','auto'])
                },
                {
                    'type': 'RandomUnderSampler',
                    'replacement' : hp.choice('replacement_RUS', [True, False]),
                    'sampling_strategy':hp.choice('sampling_strategy19',['majority','not minority','not majority','all','auto'])
                },
                {
                    'type': 'TomekLinks',
                    'sampling_strategy':hp.choice('sampling_strategy20',['majority','not minority','not majority','all','auto'])
                },
                {
                    'type': 'ClusterCentroids',
                    'estimator': hp.choice('estimator_CL',['KMeans', 'MiniBatchKMeans']),
                    'voting' : hp.choice('voting_CL',['hard', 'soft']),
                    'sampling_strategy':hp.choice('sampling_strategy21',['majority','not minority','not majority','all','auto'])
                }


            ])
        }
    ])

    return HPOspace
# In[4]:
resampler_group={'NO':'NO','SMOTE':'OVER','BorderlineSMOTE':'OVER','SMOTENC':'OVER','SMOTEN':'OVER','SVMSMOTE':'OVER','KMeansSMOTE':'OVER'
                 ,'ADASYN':'OVER','RandomOverSampler':'OVER',
                 'SMOTEENN':'COMBINE','SMOTETomek':'COMBINE',
                 'CondensedNearestNeighbour':'UNDER','EditedNearestNeighbours':'UNDER',
                 'RepeatedEditedNearestNeighbours':'UNDER','AllKNN':'UNDER',
                 'InstanceHardnessThreshold':'UNDER','NearMiss':'UNDER',
                            'NeighbourhoodCleaningRule':'UNDER','OneSidedSelection':'UNDER','RandomUnderSampler':'UNDER',
                            'TomekLinks':'UNDER','ClusterCentroids':'UNDER'}
BIG_VALUE =-1
def fscore(params_org):
    #print(params_org)
    parambk = copy.deepcopy(params_org)
    ifError =0
    global best, HPOalg,params_best, errorcount,resampler_group,randomstate,acctxt, _bestBefore,_bestAfter,Mclasstxt
    global X_train, y_train, X_test, y_test
    params= params_org['classifier']
    classifier = params.pop('name')
    xxx = params_org.pop('random_state')
    p_random_state=randomstate
    if (classifier == 'SVM'):  
        param_value= params.pop('gamma_value')
        if(params['gamma'] == "value"):
            params['gamma'] = param_value
        else:
            pass   
        clf = SVC(max_iter = 10000, cache_size= 700, random_state = p_random_state,**params)
        #max_iter=10000 and cache_size= 700 https://github.com/EpistasisLab/pennai/issues/223
        #maxvalue https://github.com/hyperopt/hyperopt-sklearn/blob/fd718c44fc440bd6e2718ec1442b1af58cafcb18/hpsklearn/components.py#L262
    elif(classifier == 'RF'):        
        clf = RandomForestClassifier(random_state = p_random_state, **params)
    elif(classifier == 'KNN'):
        p_value = params.pop('p')
        if(p_value==0):
            params['metric'] = "chebyshev"
        elif(p_value==1):
            params['metric'] = "manhattan"
        elif(p_value==2):
            params['metric'] = "euclidean"
        else:
            params['metric'] = "minkowski"
            params['p'] = p_value
        #https://github.com/hyperopt/hyperopt-sklearn/blob/fd718c44fc440bd6e2718ec1442b1af58cafcb18/hpsklearn/components.py#L302
        clf = KNeighborsClassifier(**params)
    elif(classifier == 'DTC'):        
        clf = DecisionTreeClassifier(random_state = p_random_state, **params)
    elif(classifier == 'LR'):        
        penalty_solver = params.pop('penalty_solver')
        params['penalty'] = penalty_solver.split("+")[0]
        params['solver'] = penalty_solver.split("+")[1]
        clf = LogisticRegression(random_state = p_random_state, **params)
    if Mclasstxt=='ovr':
        clf=OneVsRestClassifier(clf)
    elif Mclasstxt=='ovo':
        clf=OneVsOneClassifier(clf)
    else:
        pass
    #resampling parameter
    p_sub_params= params_org.pop('sub')
    p_sub_type = p_sub_params.pop('type')
    #sampler = p_sub_params.pop('smo_grp')
    sampler = resampler_group[p_sub_type]
    if p_sub_type not in ('EditedNearestNeighbours','RepeatedEditedNearestNeighbours','AllKNN',
                              'NearMiss','NeighbourhoodCleaningRule','TomekLinks'):
            p_sub_params['random_state']=p_random_state

    
    if 'n_neighbors' in p_sub_params:
        p_sub_params['n_neighbors']=int(p_sub_params['n_neighbors'])
    if (p_sub_type == 'SMOTE'):
        smo = SMOTE(**p_sub_params)
    elif (p_sub_type == 'ADASYN'):
        smo = ADASYN(**p_sub_params)
    elif (p_sub_type == 'BorderlineSMOTE'):
        smo = BorderlineSMOTE(**p_sub_params)
    elif (p_sub_type == 'SVMSMOTE'):
        smo = SVMSMOTE(**p_sub_params)
    elif (p_sub_type == 'SMOTENC'):
        smo = SMOTENC(**p_sub_params)
    elif (p_sub_type == 'SMOTEN'):
        smo = SMOTEN(**p_sub_params)
    elif (p_sub_type == 'KMeansSMOTE'):
        smo = KMeansSMOTE(**p_sub_params)
    elif (p_sub_type == 'RandomOverSampler'):
        smo = RandomOverSampler(**p_sub_params)
#Undersampling
    elif (p_sub_type == 'TomekLinks'):
        smo = TomekLinks(**p_sub_params)
    elif (p_sub_type == 'ClusterCentroids'):
        if(p_sub_params['estimator']=='KMeans'):
            p_sub_params['estimator']= KMeans(random_state = p_random_state)
        elif(p_sub_params['estimator']=='MiniBatchKMeans'):
            p_sub_params['estimator']= MiniBatchKMeans(random_state = p_random_state)
        smo = ClusterCentroids(**p_sub_params) 
    elif (p_sub_type == 'RandomUnderSampler'):
        smo = RandomUnderSampler(**p_sub_params)
    elif (p_sub_type == 'NearMiss'):
        smo = NearMiss(**p_sub_params)
    elif (p_sub_type == 'InstanceHardnessThreshold'):
        if(p_sub_params['estimator']=='knn'):
            p_sub_params['estimator']= KNeighborsClassifier()
        elif(p_sub_params['estimator']=='decision-tree'):
            p_sub_params['estimator']=DecisionTreeClassifier()
        elif(p_sub_params['estimator']=='adaboost'):
            p_sub_params['estimator']=AdaBoostClassifier()
        elif(p_sub_params['estimator']=='gradient-boosting'):
            p_sub_params['estimator']=GradientBoostingClassifier()
        elif(p_sub_params['estimator']=='linear-svm'):
            p_sub_params['estimator']=CalibratedClassifierCV(LinearSVC())
        elif(p_sub_params['estimator']=='random-forest'):
            p_sub_params['estimator']=RandomForestClassifier(n_estimators=100)
        smo = InstanceHardnessThreshold(**p_sub_params) 
    elif (p_sub_type == 'CondensedNearestNeighbour'):
        smo = CondensedNearestNeighbour(**p_sub_params)
    elif (p_sub_type == 'EditedNearestNeighbours'):
        smo = EditedNearestNeighbours(**p_sub_params)
    elif (p_sub_type == 'RepeatedEditedNearestNeighbours'):
        smo = RepeatedEditedNearestNeighbours(**p_sub_params) 
    elif (p_sub_type == 'AllKNN'):
        smo = AllKNN(**p_sub_params)
    elif (p_sub_type == 'NeighbourhoodCleaningRule'):
        smo = NeighbourhoodCleaningRule(**p_sub_params) 
    elif (p_sub_type == 'OneSidedSelection'):
        smo = OneSidedSelection(**p_sub_params)
#Combine
    elif (p_sub_type == 'SMOTEENN'):
        smo = SMOTEENN(**p_sub_params)
    elif (p_sub_type == 'SMOTETomek'):
        smo = SMOTETomek(**p_sub_params)
    e=''
    _results=[]
    accuracy_score1,f1_macro,f1_micro,f1_weighted,gm_macro,gm_micro,gm_weighted=0,0,0,0,0,0,0
    accuracy_score2,f1_macro2,f1_micro2,f1_weighted2,gm_macro2,gm_micro2,gm_weighted2=0,0,0,0,0,0,0
    _before,_after='',''
    
    try:        
        if(p_sub_type=='NO'):
            X_smo_train, y_smo_train = X_train, y_train 
        else:
            X_smo_train, y_smo_train = smo.fit_resample(X_train, y_train)
        y_test_pred = clf.fit(X_smo_train, y_smo_train).predict(X_test)
        y_test_pred=pd.Series(y_test_pred)
        #print(y_test_pred)
        _before,_after = cost_matrix(y_test, y_test_pred )
       
        accuracy_score1=_before['acc']
        f1_macro=_before['F1_macro']
        f1_micro=_before['F1_micro']
        f1_weighted=_before['F1_weight']
        gm_macro = _before['GM_macro']
        gm_micro =_before['GM_micro']
        gm_weighted = _before['GM_weight']
        
        accuracy_score2=_after['acc']
        f1_macro2=_after['F1_macro']
        f1_micro2=_after['F1_micro']
        f1_weighted2=_after['F1_weight']
        gm_macro2 = _after['GM_macro']
        gm_micro2 =_after['GM_micro']
        gm_weighted2 = _after['GM_weight']
        accLst={'acc1':accuracy_score1,'f1M1':f1_macro,'f1m1':f1_micro,'f1w1':f1_weighted,'gmM1':gm_macro,
                'gmm1':gm_micro,'gmw1':gm_weighted,'acc2':accuracy_score2,'f1M2':f1_macro2,'f1m2':f1_micro2,
                'f1w2':f1_weighted2,'gmM2':gm_macro2,
                'gmm2':gm_micro2,'gmw2':gm_weighted2
               }
        _mresult=accLst[acctxt]
        #print(_mresult,accuracy_score1,f1_macro,f1_micro,f1_weighted,gm_macro,gm_micro,gm_weighted)
        _results.append(_mresult)
    except Exception as eec:
        #print(eec)
        e=eec
        _mresult = BIG_VALUE
        ifError =1 
        errorcount = errorcount+1
    #gm_loss = 1 - mean_g
    abc=time.time()-starttime
    if _mresult > best:
        best = _mresult
        params_best = copy.deepcopy(parambk)
        _bestBefore,_bestAfter=_before,_after
    return {'loss': -_mresult,
            'mean': _mresult,
            'before':_before,
            'after': _after,
             
            'status': STATUS_OK,         
            # -- store other results like this
            'run_time': abc,
            'iter': iid,
            'current_best': best,
            'eval_time': time.time(),
            'classifier': classifier,
            'SamplingGrp': sampler,
            'SamplingType': p_sub_type,
            'ifError': ifError,
            'Error': e,
            'params' : parambk,
            'attachments':
                {'time_module': pickle.dumps(time.time)}
           }   



data=pd.read_csv(file,index_col=0, low_memory=False)
enc = LabelEncoder()
#######LOAD TRAIN DATA######
with open(trainIDs) as f:
    TrainIDs = [int(i) for i in f]
with open(testIDs) as f:
    TestIDs = [int(i) for i in f]
data_train=data[data.index.isin(TrainIDs)]
data_test=data[data.index.isin(TestIDs)]

X_train=data_train.iloc[:, 16:data_train.shape[1]]
X_test = data_test.iloc[:, 16:data_test.shape[1]]
y_train = data_train['NOCLASSE']
y_test = data_test['NOCLASSE']
stand=StandardScaler().fit(X_train)
X_train=np.c_[stand.transform(X_train)]
X_test=np.c_[stand.transform(X_test)]
#X = data.iloc[:, 16:data.shape[1]]
labelname = sorted(list(set(data['NOCLASSE'])))



# In[ ]:


#return X_train, X_test, y_train, y_test,labelname
# In[5]:
seeds= [seed] if seed>0 else [*range(1,11)] 
for n_init_sample in [50]:
    for randomstate in seeds: 
        print('\033[91m',HPOalg,'==',randomstate,'=== START DATASET: ', dataset, '=======', '\033[0m') 
        
        #print(X.shape)        
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=randomstate)
        space = getSP(randomstate)   
        best,params_best,_bestBefore,_bestAfter = 0,'','',''
        trials = Trials()
        starttime = time.time()
        ran_best = 0 
        best = 0        
        iid = 0
        errorcount=0
        rstate=np.random.RandomState(randomstate)
        suggest= partial(tpe.suggest, n_startup_jobs=n_init_sample)
        try:
            xOpt= fmin(fscore, space, algo=suggest, max_evals=500, trials=trials, rstate=rstate)
        except:
            print('==ERROR: RANDOM-',dataset,'===')
        runtime=time.time()-starttime

        try:
            ran_results = pd.DataFrame({'current_best': [x['current_best'] for x in trials.results],
                                        'run_time':[x['run_time'] for x in trials.results],
                                        'classifier': [x['classifier'] for x in trials.results],
                                        'SamplingGrp': [x['SamplingGrp'] for x in trials.results],                                    
                                        'SamplingType': [x['SamplingType'] for x in trials.results], 
                                        'ifError': [x['ifError'] for x in trials.results], 
                                        'Error': [x['Error'] for x in trials.results], 
                                        'loss': [x['loss'] for x in trials.results], 
                                        'mean': [x['mean'] for x in trials.results], 
                                        'before':[x['before'] for x in trials.results], 
                                        'after':[x['after'] for x in trials.results],
                                        'iteration': trials.idxs_vals[0]['classifier_type'],
                                        'params':[x['params'] for x in trials.results]})
            ran_results.to_csv(HomeFolder+'/LOGS/TPE/'+Mclasstxt+'_'+acctxt+'_'+str(n_init_sample)+'_hyperopt_'+HPOalg+'_'
                               +dataset+'_'+str(randomstate)+'_'+str(dsfold)+'.csv', 
                               index = True, header=True)
        except:
            print('ERROR: No logfile')
        finallog= HomeFolder+"/TPE_hyperopt.csv"
        if (os.path.exists(finallog)==False):
            with open(finallog, "a") as f:    
                wr = csv.writer(f, dialect='excel')
                wr.writerow(['Mclasstxt','acctxt','dataname','HPOalg','random_state','initsample','mean',
                             'before','after','params','runtime','errorcount'])
        with open(finallog, "a") as f:
            wr = csv.writer(f, dialect='excel')
            wr.writerow([Mclasstxt,acctxt,dataset,dsfold,randomstate,n_init_sample,best,_bestBefore,_bestAfter,params_best,runtime,errorcount])

