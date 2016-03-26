
# coding: utf-8

# In[1]:

import xgboost as xgb
import pandas as pd
import numpy as np
import random

# In[3]:

print("Load labels")
label_file = pd.read_csv("./train/labels_ext.csv", index_col=0)
Xtrain = pd.read_csv("./train/joint_accident_person_train_ext.csv", index_col=0)
Xtest = pd.read_csv("./test/joint_accident_person_test_ext.csv", index_col=0)

ID= Xtest['ID'].astype(np.int64)

temp = pd.merge(Xtrain, label_file, on=['ID'], how='inner')
y = temp['DRUNK_DR']

# In[4]:

# drops = ["Unnamed: 0",
#          "VE_FORMS.y","HARM_EV.y","MAN_COLL.y","RAIL","TWAY_ID",
#          "CF1","CF2","CF3",
#          "VIN","VIN_1","VIN_2","VIN_3","VIN_4","VIN_5","VIN_6","VIN_7","VIN_8",
#          "VIN_9","VIN_10","VIN_11","VIN_12","MCARR_ID",
#          "VEH_NO.y","COUNTY.y", "DAY.y","MONTH.y","HOUR.y","MINUTE.y","ROAD_FNC.y",
#          "SCH_BUS.y","MAKE.y","MAK_MOD.y","BODY_TYP.y","MOD_YEAR.y",
#          "TOW_VEH.y","SPEC_USE.y","EMER_USE.y","ROLLOVER.y","IMPACT1.y","FIRE_EXP.y",
#          "CERT_NO"
#         ]



# In[7]:

Xtrain.drop('ID', axis= 1, inplace=True)
train_y = y
Xtest.drop('ID', axis= 1, inplace=True)

# error AttributeError: 'NoneType' object has no attribute 'fillna'
# train_x.fillna(-1, axis=0, inplace= True)
# test_x.fillna(-1, axis=0, inplace= True)


# In[8]:

trainX = Xtrain
trainY = train_y.astype(int)
testX = Xtest


# In[9]:

dtrain = xgb.DMatrix(trainX, missing=-1, label=trainY)
dtest = xgb.DMatrix(testX, missing=-1)


# ### Parameter tuning
# 1. for `scale_pos_weight`, read more [here](https://github.com/dmlc/xgboost/blob/master/demo/kaggle-higgs/speedtest.py)
# 2. If you care about predicting the right probability, which in such a case, you cannot re-balance the dataset. Set parameter `max_delta_step` to a finite number (say 1) will help convergence. Read more [here](https://github.com/dmlc/xgboost/blob/master/doc/param_tuning.md)

# In[10]:

# parameter tuning
# random.seed(1024)

param = {}
param['objective'] = 'binary:logistic'

# scale weight of positive examples
# rescale weight to make it same as test set
# testsize = 543015
# weight = dtrain[:,31] * float(test_size) / len(train_y)
# param['scale_pos_weight'] = sum_wneg/sum_wpos

ratio = float(np.sum(train_y == 1)) / np.sum(train_y == 0)
param['scale_pos_weight'] = ratio

param['eta'] = 0.05
param['max_depth'] = 7
param['eval_metric'] = 'auc'
# param['silent'] = 1
param['min_child_weight'] = 1
param['subsample'] = 0.7
param['colsample_bytree'] = 0.7
param['nthread'] = 8

# Sean:: can't understand how to get dtrain[:, 31]=weight !!
# def fpreproc(dtrain, dtest, param):
#     label = dtrain.get_label()
#     ratio = float(np.sum(label == 0)) / np.sum(label==1)
#     print("ratio= ", ratio)
#     param['scale_pos_weight'] = ratio
#     wtrain = dtrain.get_weight()
#     wtest = dtest.get_weight()
#     sum_weight = sum(wtrain) + sum(wtest)
#     wtrain *= sum_weight / sum(wtrain)
#     wtest *= sum_weight / sum(wtest)
#     print("final wtrain, wtest = ", wtrain, wtest)
#     dtrain.set_weight(wtrain)
#     dtest.set_weight(wtest)
#     return (dtrain, dtest, param)

# XGBoostError: b'[13:07:13] src/metric/rank_metric.cc:36: 
# Check failed: (info.weights.size()) == (ndata) we need weight to evaluate ams'
#
# bst_cv = xgb.cv(param, dtrain, num_boost_round=50, nfold=5, metrics={'ams@0.15', 'auc'}, seed=0)
# bst_cv


# In[11]:

#bst_cv = xgb.cv(param, dtrain, num_boost_round=1000, nfold=5, seed=0)


# In[12]:

# tst = pd.DataFrame(bst_cv)
# test = tst['train-auc-mean']-tst['train-auc-std']
# test.idxmax()


# In[13]:

#num_round = test.idxmax()

num_round = 10


# In[14]:

bst = xgb.train(param, dtrain, num_round)


# In[15]:

ypred = bst.predict(dtest)


# In[18]:

ypred


# ### Averaging the votes in prediction

# In[36]:

predict_df = pd.DataFrame(data={'ID': ID, 'DRUNK_DR': ypred})
grouped_predict = predict_df.groupby('ID', as_index=False).mean()


# ### Given a prediction, create a Kaggle submission file
# 

# In[37]:

grouped_predict.to_csv('fars_submit_xgb005_production_weighted_Tian_acc_per.csv', index = False)

