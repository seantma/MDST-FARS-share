
# coding: utf-8

# # Train on accident and person data
# 

# In[1]:

import os
import pandas as pd
import numpy as np
import seaborn as sns
import sqlite3
import xgboost as xgb
from numpy import nan as NA
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score as auc


# # Load data and make labels consistent

# In[2]:

print("Load labels")
label_file = pd.read_csv("./train/labels_ext.csv", index_col=0)


# In[3]:

Xtrain = pd.read_csv("./train/joint_accident_person_train_ext.csv", index_col=0)
n_train, n_dim = Xtrain.shape
print([n_train, n_dim])


# In[4]:

temp = pd.merge(Xtrain, label_file, on=['ID'], how='inner')


# In[5]:

y = temp['DRUNK_DR'].apply(lambda x: x*1 ).values


# In[6]:

# get_ipython().magic(u'xdel temp')


# In[7]:

Xtrain.drop('ID', axis= 1, inplace=True)
print("Splitting data")


# In[8]:

Xtrain_c, Xeval, Ytrain_c, Yeval = train_test_split(Xtrain.values, y, test_size = 0.3)


# In[9]:

#get_ipython().magic(u'xdel Xtrain')
#get_ipython().magic(u'xdel label_file')


# In[10]:

# get_ipython().magic(u'whos')


# In[11]:

print("Load data into sparse matrix")
dtrain = xgb.DMatrix(data=Xtrain_c, missing = NA, label= Ytrain_c)
deval  = xgb.DMatrix(data=Xeval, missing = NA, label = Yeval)
print("Specifying the parameters ... ")


# In[15]:

param = {'max_depth': 12,
         'eta': 0.02,
         'subsample': 0.7,
         'colsample_bytree': 0.8,
         'silent': 0,
         'eval_metric': 'auc',
         'alpha': 0,
         'lambda': 1,
         'nthread': 8,
         'objective': 'binary:logistic'}


# In[16]:

watchlist = [(deval, 'eval'), (dtrain, 'train')]
num_round = 520
print("Training ... ")


# In[18]:

bst = xgb.train(param, dtrain, num_round, watchlist)


# In[19]:

print("Saving the model")
bst.save_model('./models/xgb_acc_per.model')
bst.dump_model('./models/xgb_raw_acc_per.txt')

dtrain.save_binary('./models/binary/dtrain.buffer')
deval.save_binary('./models/binary/deval.buffer')

#get_ipython().magic(u'xdel dtrain')
#get_ipython().magic(u'xdel deval')


# In[21]:

Xtest = pd.read_csv("./test/joint_accident_person_test_ext.csv", index_col=0)
ID= Xtest['ID'].astype(np.int64)
Xtest.drop('ID', axis= 1, inplace=True)
print("Load test samples")


# In[22]:

uniqueID = np.unique(ID)


# In[ ]:

uniqueID


# In[23]:

dtest= xgb.DMatrix(data=Xtest.values, missing = NA)


# In[24]:

print("Prediction")
preds = bst.predict(dtest)


# In[25]:

dtest.save_binary('./models/binary/dtest.buffer')


# In[27]:

print("submit")
predict_df = pd.DataFrame(data={'ID': ID.values, 'prob': preds})
predict_df


# In[28]:

predict_df.groupby('ID').mean()


# In[30]:

grouped_predict = predict_df.groupby('ID')
prediction = grouped_predict.mean()


# In[32]:

prediction.index


# In[34]:

submit = pd.DataFrame(data={'ID': prediction.index, 'DRUNK_DR': prediction})
submit.to_csv('fars_submit_j_acc_per_ext_1.csv', index = False)

