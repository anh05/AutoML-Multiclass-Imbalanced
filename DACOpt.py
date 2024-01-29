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
from dacopt import DACOpt, ConfigSpace, ConditionalSpace, AlgorithmChoice, IntegerParam, FloatParam, CategoricalParam
from sklearn import metrics
import glob


# In[2]:


HPOopitmizer='BO4ML'
dataset1=str(sys.argv[1])
seed=int(sys.argv[2])
acctxt= str(sys.argv[3])
Mclasstxt= str(sys.argv[4])
dsfold= int(sys.argv[5])
compare_strategy='Highest'
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

seeds= [seed] if seed>0 else [*range(1,11)] 

eta=3
isMax=True
isFair=True
_max_threads=1
n_init_sample=50
n_init_sp=n_init_sample*2 #use for DACOpt: number of samples for initial round per candidate
number_candidates=10
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


# In[3]:


def get_sp(randomstate,n_init_sample, isMax=True):
    search_space = ConfigSpace()
    con = ConditionalSpace("test")
    random_state=CategoricalParam(randomstate,'random_state')
    alg_namestr=AlgorithmChoice(['SVM','RF','KNN','DTC','LR'], 'classifier', default='SVM')    
    search_space.add_multiparameter([random_state,alg_namestr])    
    #SVM
    probability=CategoricalParam([True, False],'probability')
    C=FloatParam([0.03125 , 200],'C')
    kernel=CategoricalParam(['linear','rbf','poly', 'sigmoid'], 'kernel', default='linear')
    degree=IntegerParam([2,5],'degree')
    gamma=CategoricalParam([['auto','scale'],'value'], 'gamma', default='auto')
    gamma_value=FloatParam([3.1E-05, 8], 'gamma_value')
    coef0=FloatParam([-1,1], 'coef0')
    shrinking=CategoricalParam([True, False],'shrinking')
    tol_svm=FloatParam([1e-05, 1e-01], 'tol')
    search_space.add_multiparameter([probability,C,kernel,degree,gamma,gamma_value,coef0,shrinking,tol_svm])
    con.addMutilConditional([probability,C,kernel,degree,gamma,gamma_value,coef0,shrinking,tol_svm],alg_namestr,'SVM')
    #con.addConditional(gamma_value, gamma,'value')    
    ##RF
    n_estimators=IntegerParam([1,150],'n_estimators')
    criterion=CategoricalParam(['gini', 'entropy'],'criterion')
    max_features_RF=CategoricalParam([1, 'sqrt','log2',None],'max_features')  
    min_samples_split=IntegerParam([2, 20],'min_samples_split')
    min_samples_leaf=IntegerParam([1, 20],'min_samples_leaf')
    bootstrap=CategoricalParam([True, False],'bootstrap')
    class_weight=CategoricalParam([['balanced','balanced_subsample'],None],'class_weight')
    search_space.add_multiparameter([n_estimators,criterion,max_features_RF,min_samples_split,min_samples_leaf,
                                     bootstrap,class_weight])
    con.addMutilConditional([n_estimators,criterion,max_features_RF,min_samples_split,
                             min_samples_leaf,bootstrap,class_weight],alg_namestr,'RF')
    ###KNN
    n_neighbors_knn=IntegerParam([1,51],'n_neighbors_knn')
    weights=CategoricalParam(['uniform', 'distance'],'weights')
    algorithm=CategoricalParam(['auto', 'ball_tree', 'kd_tree', 'brute'],'algorithm')
    p=IntegerParam([0,20],'p_value')
    search_space.add_multiparameter([n_neighbors_knn,weights,algorithm,p])
    con.addMutilConditional([n_neighbors_knn,weights,algorithm,p],alg_namestr,'KNN')
    ####DTC    
    criterion_dtc=CategoricalParam(['gini', 'entropy'],'criterion_dtc')
    max_features_dtc=CategoricalParam([1, 'sqrt','log2',None],'max_features_dtc')
    max_depth=IntegerParam([2,20],'max_depth_dtc')
    min_samples_split_dtc=IntegerParam([2, 20],'min_samples_split_dtc')
    min_samples_leaf_dtc=IntegerParam([1, 20],'min_samples_leaf_dtc')
    #search_space.add_multiparameter([max_depth])
    #con.addMutilConditional([criterion,max_features_RF,min_samples_split,min_samples_leaf,max_depth],alg_namestr,"DTC")
    search_space.add_multiparameter([criterion_dtc,max_features_dtc,max_depth,min_samples_split_dtc,min_samples_leaf_dtc])
    con.addMutilConditional([criterion_dtc,max_features_dtc,max_depth,min_samples_split_dtc,min_samples_leaf_dtc],alg_namestr,"DTC")
    #####LR
    C_lr=FloatParam([0.03125 , 100],'C_LR')
    penalty_solver=CategoricalParam([['l1+liblinear','l2+liblinear'],
                                     ['l1+saga','l2+saga','elasticnet+saga','none+saga'],['l2+sag','none+sag'],
                                     ['l2+newton-cg','none+newton-cg'],['l2+lbfgs','none+lbfgs']],'penalty_solver')
    tol_lr=FloatParam([1e-05, 1e-01], 'tol_lr')
    l1_ratio=FloatParam([1e-09, 1], 'l1_ratio')
    search_space.add_multiparameter([C_lr,penalty_solver,tol_lr,l1_ratio])
    con.addMutilConditional([C_lr,penalty_solver,tol_lr,l1_ratio],alg_namestr,'LR')
    smo_type=AlgorithmChoice([[['ClusterCentroids'],['NearMiss','RandomUnderSampler'],
                                  [['CondensedNearestNeighbour','NeighbourhoodCleaningRule'],
                                  ['EditedNearestNeighbours','RepeatedEditedNearestNeighbours','AllKNN'],
                                  ['TomekLinks'],['OneSidedSelection'],['InstanceHardnessThreshold']]],
                                 [['NO'],['SMOTEENN','SMOTETomek'],[['RandomOverSampler'],['ADASYN'],
                                                                    ['SMOTE','SMOTENC','SMOTEN','BorderlineSMOTE',
                                                                     'KMeansSMOTE','SVMSMOTE']]]],'resampler')
    search_space._add_singleparameter(smo_type)
    k_neighbors_SMOTE=IntegerParam([1,10],'k_neighbors_SMOTE')
    k_neighbors_Borderline=IntegerParam([1,10],'k_neighbors_Borderline')
    m_neighbors_Borderline=IntegerParam([1,10],'m_neighbors_Borderline')
    kind=CategoricalParam(['borderline-1', 'borderline-2'],'kind')
    categorical_features=CategoricalParam([True],'categorical_features')
    k_neighbors_SMOTENC=IntegerParam([1,10],'k_neighbors_SMOTENC')
    k_neighbors_SVMSMOTE=IntegerParam([1,10],'k_neighbors_SVMSMOTE')
    m_neighbors_SVMSMOTE=IntegerParam([1,10],'m_neighbors_SVMSMOTE') 
    out_step=FloatParam([0 , 1],'out_step')   
    k_neighbors_KMeansSMOTE=IntegerParam([1,10],'k_neighbors_KMeansSMOTE')  
    cluster_balance_threshold=FloatParam([1e-2, 1],'cluster_balance_threshold')
    n_neighbors_OVER=IntegerParam([1,10],'n_neighbors_OVER')
    over_strategy= CategoricalParam(['minority','not minority','not majority','all','auto'],'sampling_strategy')
    search_space.add_multiparameter([k_neighbors_SMOTE,k_neighbors_Borderline,m_neighbors_Borderline,kind,
                                     categorical_features,k_neighbors_SMOTENC,
                                     k_neighbors_SVMSMOTE,m_neighbors_SVMSMOTE,out_step,
                                     k_neighbors_KMeansSMOTE,cluster_balance_threshold,n_neighbors_OVER, 
                                     over_strategy])
    con.addMutilConditional([k_neighbors_SMOTE,over_strategy], smo_type,'SMOTE')
    con.addMutilConditional([k_neighbors_Borderline,m_neighbors_Borderline,kind,over_strategy], smo_type,'BorderlineSMOTE')
    con.addMutilConditional([categorical_features,k_neighbors_SMOTENC,over_strategy], smo_type,'SMOTENC')
    con.addMutilConditional([k_neighbors_SVMSMOTE,m_neighbors_SVMSMOTE,out_step,over_strategy], smo_type,'SVMSMOTE')
    con.addMutilConditional([k_neighbors_KMeansSMOTE,cluster_balance_threshold,over_strategy], smo_type,'KMeansSMOTE')
    con.addMutilConditional([n_neighbors_OVER,over_strategy], smo_type,'ADASYN')
    con.addMutilConditional([n_neighbors_OVER,over_strategy], smo_type,'SMOTEN')
    con.addConditional(over_strategy, smo_type,'RandomOverSampler')
    n_neighbors_UNDER50=IntegerParam([1,50],'n_neighbors_CNN')
    n_seeds_S=IntegerParam([1,50],'n_seeds_S_CNN')
    n_neighbors_UNDER1=IntegerParam([1,20],'n_neighbors_UNDER1')
    kind_sel1=CategoricalParam(['all','mode'],'kind_sel1')
    n_neighbors_UNDER2=IntegerParam([1,20],'n_neighbors_UNDER2')
    kind_sel2=CategoricalParam(['all','mode'],'kind_sel2')
    n_neighbors_UNDER3=IntegerParam([1,20],'n_neighbors_UNDER3')
    kind_sel3=CategoricalParam(['all','mode'],'kind_sel3')
    allow_minority=CategoricalParam([True, False],'allow_minority')
    estimator_IHT=CategoricalParam(['knn', 'decision-tree', 'adaboost','gradient-boosting','linear-svm',None],'estimator_IHT')
    cv_under=IntegerParam([2,20],'cv')
    version=CategoricalParam([1,2,3],'version')
    n_neighbors_UNDER4=IntegerParam([1,20],'n_neighbors_UNDER4')
    n_neighbors_ver3=IntegerParam([1,20],'n_neighbors_ver3')
    n_neighbors_UNDER5=IntegerParam([1,20],'n_neighbors_UNDER5')
    threshold_cleaning_NCR=FloatParam([0 , 1],'threshold_cleaning')
    n_neighbors_UNDER6=IntegerParam([1,20],'n_neighbors_UNDER6')
    n_seeds_S_under=IntegerParam([1,20],'n_seeds_S')
    replacement=CategoricalParam([True, False],'replacement')
    estimator_CL=CategoricalParam(['KMeans', 'MiniBatchKMeans'],'estimator')
    voting_CL=CategoricalParam(['hard', 'soft'],'voting')
    under_strategy= CategoricalParam(['majority','not minority','not majority','all','auto'],'sampling_strategy_under')
    search_space.add_multiparameter([n_neighbors_UNDER50,n_seeds_S,n_neighbors_UNDER1,kind_sel1,
                                     n_neighbors_UNDER2,kind_sel2,n_neighbors_UNDER3,kind_sel3,                                     
                                     allow_minority,estimator_IHT,cv_under,version,n_neighbors_UNDER4,n_neighbors_ver3,
                                     n_neighbors_UNDER5,threshold_cleaning_NCR,n_neighbors_UNDER6,n_seeds_S_under,
                                     replacement,estimator_CL,voting_CL,under_strategy
                                    ])
    con.addMutilConditional([estimator_CL,voting_CL,under_strategy], smo_type,'ClusterCentroids')
    con.addMutilConditional([n_neighbors_UNDER50, n_seeds_S,under_strategy], smo_type,'CondensedNearestNeighbour')
    con.addMutilConditional([n_neighbors_UNDER1, kind_sel1,under_strategy], smo_type,'EditedNearestNeighbours')
    con.addMutilConditional([n_neighbors_UNDER2, kind_sel2,under_strategy], smo_type,'RepeatedEditedNearestNeighbours')
    con.addMutilConditional([n_neighbors_UNDER3, kind_sel3, allow_minority,under_strategy], smo_type,'AllKNN')
    con.addMutilConditional([estimator_IHT, cv_under,under_strategy], smo_type,'InstanceHardnessThreshold')
    con.addMutilConditional([version,n_neighbors_UNDER4,n_neighbors_ver3,under_strategy], smo_type,'NearMiss')
    con.addMutilConditional([n_neighbors_UNDER5,threshold_cleaning_NCR,under_strategy], smo_type,'NeighbourhoodCleaningRule')
    con.addMutilConditional([n_neighbors_UNDER6,n_seeds_S_under,under_strategy], smo_type,'OneSidedSelection')
    con.addMutilConditional([replacement,under_strategy], smo_type,'RandomUnderSampler')
    con.addConditional(under_strategy, smo_type,'TomekLinks')
    return search_space, con


def fscore(params_org):
    #print(params_org)
    parambk = copy.deepcopy(params_org)
    ifError =0
    global best, HPOalg,params_best, errorcount,resampler_group,randomstate,acctxt,Mclasstxt
    global X_train, y_train, X_test, y_test, iid,_bestBefore
    iid=iid+1
    #global best, HPOalg,params_best, errorcount,resampler_group,_idxi,_error, X,y
    #self._idxi=int(self._idxi)+1
    p_random_state = int(params_org.pop('random_state'))
    params= params_org['classifier']
    classifier = params.pop('name')    
    p_sub_params= params_org.pop('resampler')
    p_sub_type = p_sub_params.pop('name')
    #if p_sub_type == 'SMOTEN':
    #    print(parambk) 
    sampler = resampler_group[p_sub_type]
    if (classifier == 'SVM'):  
        gamma_value= params.pop('gamma_value')
        if(params['gamma'] == "value"):
            params['gamma'] = gamma_value
        #print(params)
        clf = SVC(max_iter = 10000, cache_size= 700, random_state = p_random_state,**params)
        #max_iter=10000 and cache_size= 700 https://github.com/EpistasisLab/pennai/issues/223
        #maxvalue https://github.com/hyperopt/hyperopt-sklearn/blob/fd718c44fc440bd6e2718ec1442b1af58cafcb18/hpsklearn/components.py#L262
    elif(classifier == 'RF'):        
        clf = RandomForestClassifier(random_state = p_random_state, **params)
    elif(classifier == 'KNN'):
        p_value = params.pop('p_value')
        params['n_neighbors']= params.pop('n_neighbors_knn')
        #print(self._idxi,'KNN',params)
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
        if 'criterion_dtc' in params:
            params['criterion']= params.pop('criterion_dtc')
            params['max_features']= params.pop('max_features_dtc')
            params['min_samples_split']= params.pop('min_samples_split_dtc')
            params['min_samples_leaf']= params.pop('min_samples_leaf_dtc')
        params['max_depth']= params.pop('max_depth_dtc')
        clf = DecisionTreeClassifier(random_state = p_random_state, **params)
    elif(classifier == 'LR'):
        if 'C_LR' in params:
            params['C']= params.pop('C_LR')
        if 'tol_lr' in params:
            params['tol']= params.pop('tol_lr')
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
    #resGroup=AlgorithmChoice(["NO","OVER","COMBINE","UNDER"],"ResGroup")
    notuse = ['x_no', 'x_ROS', 'x_COM1', 'x_COM2', 'x_TML', 'random_state_X']
    tobedel = [i for i in notuse if i in p_sub_params]
    for x in tobedel:
        xxxx=p_sub_params.pop(x)
    if p_sub_type not in ('EditedNearestNeighbours','RepeatedEditedNearestNeighbours','AllKNN',
                          'NearMiss','NeighbourhoodCleaningRule','TomekLinks'):
        p_sub_params['random_state']=p_random_state
    kind_sel_sets=['kind_sel1','kind_sel2','kind_sel3']
    kind_sel_value=[i for i in kind_sel_sets if i in p_sub_params]
    if len(kind_sel_value):
        p_sub_params['kind_sel']= p_sub_params.pop(kind_sel_value[0])

    k_neighbors=['k_neighbors_SMOTE','k_neighbors_Borderline','k_neighbors_SMOTENC','k_neighbors_SVMSMOTE','k_neighbors_KMeansSMOTE']
    k_value=[i for i in k_neighbors if i in p_sub_params]
    if len(k_value)>0:        
        p_sub_params['k_neighbors']= p_sub_params.pop(k_value[0])
    m_neighbors=['m_neighbors_Borderline','m_neighbors_SVMSMOTE']
    m_value=[i for i in m_neighbors if i in p_sub_params]
    if len(m_value)>0:
        p_sub_params['m_neighbors']= int(p_sub_params.pop(m_value[0]))

    n_neighbors=['n_neighbors_CNN','n_neighbors_UNDER1','n_neighbors_UNDER2','n_neighbors_UNDER3',
                 'n_neighbors_UNDER4','n_neighbors_UNDER5','n_neighbors_UNDER6','n_neighbors_OVER','x_neighbors_OVER']
    n_value=[i for i in n_neighbors if i in p_sub_params]
    if len(n_value)>0:
        p_sub_params['n_neighbors']= int(p_sub_params.pop(n_value[0]))
    if 'sampling_strategy_under' in p_sub_params:
        p_sub_params['sampling_strategy']= p_sub_params.pop('sampling_strategy_under')
    if 'estimator_IHT' in p_sub_params:
        p_sub_params['estimator']= p_sub_params.pop('estimator_IHT')
    
    if (p_sub_type == 'SMOTE'):
        smo = SMOTE(**p_sub_params)
    elif (p_sub_type == 'ADASYN'):
        #print(p_sub_type,p_sub_params)
        if 'k_neighbors' in p_sub_params:
            p_sub_params['n_neighbors']=int(p_sub_params.pop('k_neighbors'))
        smo = ADASYN(**p_sub_params)
    elif (p_sub_type == 'BorderlineSMOTE'):
        smo = BorderlineSMOTE(**p_sub_params)
    elif (p_sub_type == 'SVMSMOTE'):
        smo = SVMSMOTE(**p_sub_params)
    elif (p_sub_type == 'SMOTENC'):
        p_sub_params['categorical_features']=True
        smo = SMOTENC(**p_sub_params)
    elif (p_sub_type == 'SMOTEN'):
        #print(parambk)
        if 'n_neighbors' in p_sub_params:
            p_sub_params['k_neighbors']=int(p_sub_params.pop('n_neighbors'))
        smo = SMOTEN(**p_sub_params)
    elif (p_sub_type == 'KMeansSMOTE'):
        #print(p_sub_params)
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
        #p_sub_params['n_neighbors']= int(p_sub_params.pop('n_neighbors_CNN'))
        if 'n_seeds_S_CNN' in p_sub_params:
            p_sub_params['n_seeds_S']= p_sub_params.pop('n_seeds_S_CNN')        
        smo = CondensedNearestNeighbour(**p_sub_params)
    elif (p_sub_type == 'EditedNearestNeighbours'):
        smo = EditedNearestNeighbours(**p_sub_params)
    elif (p_sub_type == 'RepeatedEditedNearestNeighbours'):
        smo = RepeatedEditedNearestNeighbours(**p_sub_params) 
    elif (p_sub_type == 'AllKNN'):
        #print(self._idxi,'ALLKNN',p_sub_params)
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
    e,_before='',''
    _results=[]
    status=STATUS_OK
    try:
        if(p_sub_type=='NO'):
            X_smo_train, y_smo_train = X_train, y_train 
        else:
            X_smo_train, y_smo_train = smo.fit_resample(X_train, y_train)
        y_test_pred = clf.fit(X_smo_train, y_smo_train).predict(X_test)
        y_test_pred=pd.Series(y_test_pred)
        _before = accuracy_report(y_test, y_test_pred )
        #_results.append(_mresult)
        accuracy_score1=_before['acc']
        f1_macro=_before['F1_macro']
        f1_micro=_before['F1_micro']
        f1_weighted=_before['F1_weight']
        gm_macro = _before['GM_macro']
        gm_micro =_before['GM_micro']
        gm_weighted = _before['GM_weight']
        accLst={'acc1':accuracy_score1,'f1M1':f1_macro,'f1m1':f1_micro,'f1w1':f1_weighted,'gmM1':gm_macro,
                'gmm1':gm_micro,'gmw1':gm_weighted}
        _mresult=accLst[acctxt]
        #print(_mresult,accuracy_score1,f1_macro,f1_micro,f1_weighted,gm_macro,gm_micro,gm_weighted)
        _results.append(_mresult)
    except Exception as eec:            
        e=eec
        #print (e,parambk )
        _mresult = BIG_VALUE
        ifError =1 
        errorcount +=1
    #gm_loss = 1 - mean_g
    abc=time.time()-starttime
    #print(_mresult,abc)
    if _mresult >best:
        #print('***',_mresult,'***',classifier,'--',p_sub_type,parambk)
        best = _mresult
        params_best = copy.deepcopy(parambk)
        _bestBefore=_before

    return {'loss': -_mresult,
            'mean': _mresult,
            'status': status,   
            'before':_before,
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
'''y = data['NOCLASSE']
X = StandardScaler().fit_transform(X)
X = np.c_[X]    '''
labelname = sorted(list(set(data['NOCLASSE'])))

def fscore1(param):
    print(param)
    return {'loss': -np.random.uniform(0,1),'status': STATUS_OK,}
resampler_group={'NO':'NO','SMOTE':'OVER','BorderlineSMOTE':'OVER','SMOTENC':'OVER','SMOTEN':'OVER','SVMSMOTE':'OVER','KMeansSMOTE':'OVER'
                 ,'ADASYN':'OVER','RandomOverSampler':'OVER',
                 'SMOTEENN':'COMBINE','SMOTETomek':'COMBINE',
                 'CondensedNearestNeighbour':'UNDER','EditedNearestNeighbours':'UNDER',
                 'RepeatedEditedNearestNeighbours':'UNDER','AllKNN':'UNDER',
                 'InstanceHardnessThreshold':'UNDER','NearMiss':'UNDER',
                            'NeighbourhoodCleaningRule':'UNDER','OneSidedSelection':'UNDER','RandomUnderSampler':'UNDER',
                            'TomekLinks':'UNDER','ClusterCentroids':'UNDER'}
BIG_VALUE =-1

if 1==1:
    for randomstate in seeds:  
        print('\033[91m',HPOalg,'==Random Seed:',randomstate,'=== START DATASET: ', dataset, '=======', '\033[0m')
        best,params_best,_bestBefore = 0,'',''     
        iid = 0
        errorcount=0
        rstate=np.random.RandomState(randomstate)
        search_space,con=get_sp(randomstate,n_init_sample,isMax=isMax)
        trials = Trials()
        #exp=Exsupport(file,dataset,randomstate,n_init_sample)
        opt = DACOpt(search_space, fscore,conditional=con,hpo_prefix='name',isDaC=False,
                    compare_strategy=compare_strategy,
                   HPOopitmizer=HPOopitmizer,random_seed=randomstate,
                   max_eval=500,  n_init_sample=n_init_sample,number_candidates=number_candidates,n_init_sp=n_init_sp,
                   max_threads=_max_threads,isFlatSetting=False)
        starttime = time.time()
        xopt, fopt, _, eval_count = opt.run()
        runtime=time.time()-starttime
        #print(randomstate,xopt, 1-fopt, _, eval_count,runtime,exp.errorcount)
        try:
            ran_results = pd.DataFrame({'current_best': [x['current_best'] for x in opt.opt.results.values()],
                                       'run_time':[x['run_time'] for x in opt.opt.results.values()],
                                       'classifier': [x['classifier'] for x in opt.opt.results.values()],
                                       'SamplingGrp': [x['SamplingGrp'] for x in opt.opt.results.values()],
                                       'SamplingType': [x['SamplingType'] for x in opt.opt.results.values()], 
                                       'ifError': [x['ifError'] for x in opt.opt.results.values()], 
                                       'Error': [x['Error'] for x in opt.opt.results.values()], 
                                       'loss': [x['loss'] for x in opt.opt.results.values()], 
                                       'iteration': [i for i in opt.opt.results.keys()],
                                       'before':[x['before'] for x in opt.opt.results.values()],
                                       'params':[x['params'] for x in opt.opt.results.values()]})
            ran_results.to_csv(HomeFolder+'/LOGS/DACOpt/'+Mclasstxt+'_'+compare_strategy+'_'+str(n_init_sample)+'_'+
                               str(number_candidates)+'_'+HPOopitmizer+'_'+dataset+'_'+str(randomstate)+'_'+str(dsfold)+'.csv',
                               index = True, header=True)
        except:
            print('ERROR: No logfile')
        finallog= HomeFolder+"/DACOpt.csv"
        if (os.path.exists(finallog)==False):
            with open(finallog, "a") as f:    
                wr = csv.writer(f, dialect='excel')
                wr.writerow(['dataname','Mclasstxt','HPOalg','HPOopitmizer','random_state','initsample','number_candidates',
                             'compare_strategy','max_threads','eta','mean',
                             'params','bestBefore','runtime','errorcount','isMax','isFair'])
        with open(finallog, "a") as f:
            wr = csv.writer(f, dialect='excel')
            wr.writerow([dataset,Mclasstxt,dsfold,HPOopitmizer,randomstate,n_init_sample,number_candidates,compare_strategy,
                         _max_threads,eta,-fopt,xopt,_bestBefore,runtime,errorcount,isMax, isFair])

