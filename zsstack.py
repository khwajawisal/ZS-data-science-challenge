import pandas as pd
import numpy as np
import seaborn as sns
from catboost import CatBoostClassifier
#sys.setrecursionlimit(2000)
import sys
sys.setrecursionlimit(2000)

df=pd.read_csv(r'datazs.csv')

df1=df.drop('Unnamed: 0',axis=1)
df1

df1['type_of_combined_shot'].fillna(df1['type_of_shot'],inplace=True)

df1.isnull().sum()
df1.drop('type_of_shot',axis=1,inplace=True)

#df1['power_of_shot'].fillna(df1['power_of_shot'].mean,inplace=True)
#df1['distance_of_shot'].fillna(df1['distance_of_shot'].mean,inplace=True)
df1.isnull().sum()

df1['area_of_shot'].fillna(method='ffill', inplace = True)
df1['shot_basics'].fillna(method='ffill', inplace = True)
df1['range_of_shot'].fillna(method='ffill', inplace = True)
df1['game_season'].fillna(method='ffill', inplace = True)
df1['home/away'].fillna(method='ffill', inplace = True)
df1['team_name'].fillna(method='ffill', inplace = True)
df1['remaining_min'].fillna(method='ffill', inplace = True)
df1['remaining_sec'].fillna(method='ffill', inplace = True)
df1['power_of_shot'].fillna(method='ffill', inplace = True)
df1['distance_of_shot'].fillna(method='ffill', inplace = True)
df1['date_of_game'].fillna(method='ffill', inplace = True)
df1['location_x'].fillna(method='ffill', inplace = True)
df1['location_y'].fillna(method='ffill', inplace = True)
df1['knockout_match'].fillna(method='ffill', inplace = True)
df1['home/away'].fillna(method='ffill', inplace = True)
#df1['lat/lng'].fillna(met)

df2=df1

df2['lat/lng'].fillna(df['lat/lng'].mean,inplace =True)
df2

df2.isnull().sum()

df3=df2.drop(['match_event_id','remaining_min.1','power_of_shot.1','knockout_match.1','remaining_sec.1','distance_of_shot.1'],axis=1)

df3

df3.isnull().sum()

test_set = df3[df3.is_goal.isnull()]

#test_set.drop('is_goal',axis=1,inplace=True)
#data = test_set[test_set['shot_id_number'].notnull()]

submission=test_set[['shot_id_number','is_goal']]
test_set.columns
#test_set


test_set=test_set.drop('is_goal',axis=1)

test_set=test_set.drop('shot_id_number',axis=1)

test_set=test_set.drop(['match_id','team_id','team_name'],axis=1)

test_set.columns
submission.columns

df3.dropna(subset=['is_goal'],inplace=True)

len(df3)

len(test_set)
y=df3['is_goal']
len(y)

df3.drop(['shot_id_number','is_goal'],axis=1,inplace=True)

df3.columns

df3=df3.drop(['match_id','team_id','team_name'],axis=1)

df3.columns

#test_set.drop('is_goal',axis=1,inplace=True)
test_set.columns

X_train, X_test, y_train, y_test = train_test_split(df3, y, test_size = 0.25, random_state = 0)

model = CatBoostClassifier(iterations=1500, learning_rate=0.08, l2_leaf_reg=3.5, depth=9, rsm=0.98, loss_function= 'Logloss',eval_metric='MAE',use_best_model=True,random_seed=42)

df3.columns

cate_features_index = np.where(df3.dtypes != float)
myarray= np.asarray(cate_features_index)
myarray=myarray.astype(int)
myarray.dtype

#model.fit(X_train,y_train,)
#model.fit(X_train,y_train,cat_features=myarray,eval_set=(X_test,y_test))

df1.isnull().sum()

dfenc=df1.drop(['team_id','match_id','match_event_id','remaining_min.1','power_of_shot.1','knockout_match.1','remaining_sec.1','distance_of_shot.1'],axis=1)

dfenc.columns

len(dfenc)

sample_train = dfenc[['area_of_shot','is_goal']]

## Mean encoding 
x = sample_train.groupby(['area_of_shot'])['is_goal'].sum().reset_index()
x = x.rename(columns={"is_goal" : "area_shot_sum"})

y = sample_train.groupby(['area_of_shot'])['is_goal'].count().reset_index()
y = y.rename(columns={"is_goal" : "is_goal_count"})

z = pd.merge(x,y,on = 'area_of_shot',how = 'inner')
z['Target_Encoded_over_areashot'] = z['area_shot_sum']/z['is_goal_count']
#z.head()
#Title	Title_Survived_sum	Title_Survived_count	

z = z[['area_of_shot','Target_Encoded_over_areashot']]

sample_train = pd.merge(sample_train,z,on = 'area_of_shot',how = 'left')
sample_train.isnull().sum()

sample_train.drop('area_of_shot',inplace=True,axis=1)


dfenc['area_of_shot']=sample_train['Target_Encoded_over_areashot']

sample_train = dfenc[['shot_basics','is_goal']]

## Mean encoding 
x = sample_train.groupby(['shot_basics'])['is_goal'].sum().reset_index()
x = x.rename(columns={"is_goal" : "shot_basics_sum"})

y = sample_train.groupby(['shot_basics'])['is_goal'].count().reset_index()
y = y.rename(columns={"is_goal" : "is_goal_count1"})

z = pd.merge(x,y,on = 'shot_basics',how = 'inner')
z['Target_Encoded_over_shotbasics'] = z['shot_basics_sum']/z['is_goal_count1']
z
#Title	Title_Survived_sum	Title_Survived_count	

sample_train = pd.merge(sample_train,z,on = 'shot_basics',how = 'left')
sample_train.head()



dfenc['shot_basics']=sample_train['Target_Encoded_over_shotbasics']

dfenc.isnull().sum()

sample_train1 = dfenc[['range_of_shot','is_goal']]

## Mean encoding 
x1 = sample_train1.groupby(['range_of_shot'])['is_goal'].sum().reset_index()
x1 = x1.rename(columns={"is_goal" : "range_sum"})

y1 = sample_train1.groupby(['range_of_shot'])['is_goal'].count().reset_index()
y1 = y1.rename(columns={"is_goal" : "is_goal_count1"})

z1 = pd.merge(x1,y1,on = 'range_of_shot',how = 'inner')
z1['Target_Encoded_over_range'] = z1['range_sum']/z1['is_goal_count1']
z1
#Title	Title_Survived_sum	Title_Survived_count	

z1 = z1[['range_of_shot','Target_Encoded_over_range']]

sample_train1 = pd.merge(sample_train1,z1,on = 'range_of_shot',how = 'left')
sample_train1.head()

dfenc['range_of_shot']=sample_train1['Target_Encoded_over_range']

sample_train1 = dfenc[['game_season','is_goal']]

## Mean encoding 
x1 = sample_train1.groupby(['game_season'])['is_goal'].sum().reset_index()
x1 = x1.rename(columns={"is_goal" : "season_sum"})

y1 = sample_train1.groupby(['game_season'])['is_goal'].count().reset_index()
y1 = y1.rename(columns={"is_goal" : "is_goal_count1"})

z1 = pd.merge(x1,y1,on = 'game_season',how = 'inner')
z1['Target_Encoded_over_season'] = z1['season_sum']/z1['is_goal_count1']
z1
#Title	Title_Survived_sum	Title_Survived_count	

z1 = z1[['game_season','Target_Encoded_over_season']]

sample_train1 = pd.merge(sample_train1,z1,on = 'game_season',how = 'left')
sample_train1.head()

dfenc['game_season']=sample_train1['Target_Encoded_over_season']

sample_train1 = dfenc[['type_of_combined_shot','is_goal']]

## Mean encoding 
x1 = sample_train1.groupby(['type_of_combined_shot'])['is_goal'].sum().reset_index()
x1 = x1.rename(columns={"is_goal" : "shot_sum"})

y1 = sample_train1.groupby(['type_of_combined_shot'])['is_goal'].count().reset_index()
y1 = y1.rename(columns={"is_goal" : "is_goal_count1"})

z1 = pd.merge(x1,y1,on = 'type_of_combined_shot',how = 'inner')
z1['Target_Encoded_over_shot'] = z1['shot_sum']/z1['is_goal_count1']
z1
#Title	Title_Survived_sum	Title_Survived_count	

z1 = z1[['type_of_combined_shot','Target_Encoded_over_shot']]

sample_train1 = pd.merge(sample_train1,z1,on = 'type_of_combined_shot',how = 'left')
sample_train1.head()

dfenc['type_of_combined_shot']=sample_train1['Target_Encoded_over_shot']

dfenc

dff=dfenc.drop(['team_name','date_of_game','lat/lng'],axis=1)

dff.head()

submi_set=dff[dff.is_goal.isnull()]

submi_set.drop(['shot_id_number','is_goal'],axis=1,inplace=True)

submi_set.head()

len(dff)

len(submi_set)

len(submission)

dff.dropna(subset=['is_goal'],inplace=True)

dff.drop('shot_id_number',axis=1,inplace=True)

y=dff['is_goal']
len(y)

dff.drop('is_goal',axis=1,inplace=True)

dff.columns

submi_set.columns

from scipy import stats
#dfout=dff[(np.abs(stats.zscore(dfenc)) < 3).all(axis=1)]

subout=submi_set[(np.abs(stats.zscore(submi_set)) < 3).all(axis=1)]

len(dfout)

len(subout)

dff

X_train1, X_test1, y_train1, y_test1 = train_test_split(dff, y, test_size = 0.35, random_state = 0)

model = CatBoostClassifier(iterations=1500, learning_rate=0.03, l2_leaf_reg=16, depth=10, rsm=0.25, loss_function= 'Logloss', eval_metric='AUC',use_best_model=True,random_seed=42)

#model.fit(X_train,y_train,)
model.fit(X_train1,y_train1,eval_set=(X_test1,y_test1))

y_predcat=model.predict_proba(X_test1)
ypredcat_test=model.predict_proba(submi_set)
ypredcat_test= ypredcat_test[:,1]
y_predcat=y_predcat[:,1]
ypredcat_test

#dfstack['catpred']=pd.DataFrame(y_predcat[:])
dftest1['catpred']=pd.DataFrame(ypredcat_test[:])
#dftest.drop('0',axis=1,inplace=Tru
#dfstack.rename(columns={0:'catpred1'})

dfstack=dfstack[['catpred','mlppred']]

dfstack



from sklearn.feature_selection import RFECV
#using rfecv from feature_selection to see the fetaure importances to optimize our dataset
from sklearn.ensemble import RandomForestClassifier
clf_rf_4 =CatBoostClassifier(iterations=500, learning_rate=0.08, l2_leaf_reg=3.5, depth=9, rsm=0.98, loss_function= 'Logloss', eval_metric='Logloss',random_seed=42)
rfecv = RFECV(estimator=clf_rf_4, step=1, cv=4,scoring='accuracy')   #5-fold cross-validation
rfecv = rfecv.fit(X_train, y_train)

print('Optimal number of features :', rfecv.n_features_)

print('Best features :', X_train.columns[rfecv.support_])

dfbest=dff[['distance_of_shot','type_of_combined_shot']]

X_train1, X_test1, y_train1, y_test1 = train_test_split(dfbest, y, test_size = 0.25, random_state = 0)

model1 = CatBoostClassifier(iterations=4500, learning_rate=0.02, l2_leaf_reg=18.5, depth=12, rsm=0.2, loss_function= 'Logloss', eval_metric='AUC',use_best_model=True,random_seed=42)



#model.fit(X_train,y_train,)
model1.fit(X_train1,y_train1,eval_set=(X_test1,y_test1))

#yprob=model.predict(X_test)
yprob1=model1.predict(X_test1)

from sklearn.metrics import accuracy_score
#ac_2 = accuracy_score(y_test,yprob)
ac_1 = accuracy_score(y_test1,yprob1)
#print('Accuracy with all features: ',ac_2)
print('Accuracy with less features: ',ac_1)

test=submi_set[['location_x','location_y','distance_of_shot','shot_basics','type_of_combined_shot','remaining_sec']]

yprobab=model1.predict_proba(test)

preds=np.around(yprobab, decimals = 1)
preds= preds[:,1]

submission['is_goal']=preds
submission.to_csv('nnn.csv',index=False)

sample=pd.read_csv('sample11.csv')

merged=submission.merge(sample,on='shot_id_number')

merged.to_csv('newsub1.csv',index=False)

from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
dff.columns

scaler = MinMaxScaler()
# fit and transform in one step
dffn = scaler.fit_transform(dff)
dte = scaler.fit_transform(submi_set)

from keras.layers import advanced_activations,LeakyReLU
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

X_train2, X_test2, y_train2, y_test2 = train_test_split(dffn, y, test_size = 0.35, random_state = 0)

model_mlp = Sequential()
model_mlp.add(Dense(26, input_dim=13))
model_mlp.add(LeakyReLU(alpha=0.1))
model_mlp.add(Dense(36))
model_mlp.add(LeakyReLU(alpha=0.1))
model_mlp.add(Dense(20))
model_mlp.add(LeakyReLU(alpha=0.1))
#model_mlp.add(Dense(2))
#model_mlp.add(LeakyReLU(alpha=0.1))
model_mlp.add(Dense(12))
model_mlp.add(LeakyReLU(alpha=0.1))
model_mlp.add(Dense(1,activation='sigmoid'))
# Compile model
model_mlp.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])
# Fit the model
#model_mlp.fit(X_train2,y_train2, epochs=50, batch_size=15)
# evaluate the model
#model_mlp = model.evaluate(X_test2,y_test2)
#3print("\n%s: %.2f%%" % (model_,.metrics_names[1], scores[1]*100))
#y_pred= model.predict(x_test, verbose=1)

model_mlp.fit(X_train2,y_train2, epochs=50, batch_size=10)

y_predmlp= model_mlp.predict(X_test2, verbose=1)
dfstack['mlppred']=pd.DataFrame(y_predmlp[:])


mlpsub=model_mlp.predict(dte,verbose=1)

y_predmlp=np.around(y_predmlp, decimals = 0)

dftest1['mlppred']=pd.DataFrame(mlpsub[:])
dftest1
dfstack

import lightgbm as lgb

d_train = lgb.Dataset(X_trainf, label= y_trainf)

params = {}
params['learning_rate']= 0.03
params['boosting_type']='gbdt'
params['objective']='binary'
params['metric']='accuarcy'
params['sub_feature']=0.4
params['num_leaves']= 20
params['min_data']=80
params['max_depth']=20


clf= lgb.train(params, d_train, 1000)
#len(y_pred)

y_predlgb = clf.predict(X_testf)

#convert into binary values



#y_predtest         


for i in range(0,len(y_predlgb)):
    if (y_predlgb[i] >= 0.5):
        y_predlgb[i] = 1
    else:
        y_predlgb[i] =0


ac_1 = accuracy_score(y_testf,y_predlgb)
#print('Accuracy with all features: ',ac_2)

print('Accuracy with less features: ',ac_1)

dfstack['lgbpred']=pd.DataFrame(y_predlgb[:])
dftest1['lgbpredt']=pd.DataFrame(y_predtest[:])
dfstack

import xgboost as xgb
from xgboost import plot_importance

params = {
    'objective': 'binary:logistic',
    'gamma': 0.1,
    'max_depth': 3,
    'lambda': 5,
    'subsample': 0.6,
    'colsample_bytree': 0.7,
    'min_child_weight': 5,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

num_round = 300

dtrain = xgb.DMatrix(X_train2, label=y_train2)
dtest = xgb.DMatrix(X_test2, label=y_test2)
dt=xgb.DMatrix(submi_set.values)
watchlist = [(dtrain, 'train'), (dtest, 'test')]
bst = xgb.train(params, dtrain, num_round, watchlist)


y_pred_xgb = bst.predict(dt)

y_pred_xgb

submi_set.columns

dff.columns

rdmf = RandomForestClassifier(n_estimators=1000, criterion='entropy')
rdmf.fit(X_train2, y_train2)

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

y_predrf=rdmf.predict_proba(X_test2)
rftest=rdmf.predict_proba(submi_set)

#y_predrf=y_predrf[:,1]
dfstack['rfpred']=pd.DataFrame(y_predrf[:,1])
dftest1['rfpred']=pd.DataFrame(rftest[:,1])
dftest1

ac_1 = accuracy_score(y_test2,y_predrf)
#print('Accuracy with all features: ',ac_2)

print('Accuracy with less features: ',ac_1)

ada=AdaBoostClassifier

ada = AdaBoostClassifier(random_state=1,n_estimators=1000,learning_rate=0.02)
ada.fit(X_train2, y_train2)

y_predada = clf.predict(X_test2)
#print('Accuracy: {}'.format(accuracy_score(y_predada, y_test2)))

dfstack['adapred']=pd.DataFrame(y_predada[:])

for i in range(0,len(y_predada)):
    if (y_predada[i] >= 0.5):
        y_predada[i] = 1
    else:
        y_predada[i] =0

y_predada1 = clf.predict(dte)

dftest1['adapred']=pd.DataFrame(y_predada1[:])

dfstack.head()

dftest1.head()

yfin=y_test2

len(dftest1)

X_trainf, X_testf, y_trainf, y_testf = train_test_split(dfstack, yfin, test_size = 0.25, random_state = 0)

rdmfst = RandomForestClassifier(n_estimators=1000, criterion='entropy')
rdmfst.fit(X_trainf, y_trainf)

yp=rdmfst.predict(X_testf)

ac_2 = accuracy_score(y_testf,yp)
print('Accuracy is: ',ac_2)

model1 = CatBoostClassifier(iterations=4500, learning_rate=0.04, l2_leaf_reg=18.5, depth=6, rsm=0.2, loss_function= 'Logloss', eval_metric='AUC',use_best_model=True,random_seed=42)

#model.fit(X_train,y_train,)
model1.fit(X_trainf,y_trainf,eval_set=(X_testf,y_testf))

predi=model1.predict(X_testf)

predi

ac_2 = accuracy_score(y_testf,predi)
print('Accuracy is: ',ac_2)

subm=model.predict_proba(submi_set)
subm=np.around(subm,decimals=1)

submission['is_goal']=subm[:,1]

submission.to_csv('stack.csv',index=False)

model_mlp = Sequential()
model_mlp.add(Dense(14, input_dim=5))
model_mlp.add(LeakyReLU(alpha=0.1))
model_mlp.add(Dense(24))
model_mlp.add(LeakyReLU(alpha=0.1))
model_mlp.add(Dense(48))
model_mlp.add(LeakyReLU(alpha=0.1))
model_mlp.add(Dense(22))
model_mlp.add(LeakyReLU(alpha=0.1))
model_mlp.add(Dense(12))
model_mlp.add(LeakyReLU(alpha=0.1))
model_mlp.add(Dense(1,activation='sigmoid'))
# Compile model
model_mlp.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])
# Fit the model
#model_mlp.fit(X_train2,y_train2, epochs=50, batch_size=15)
# evaluate the model
#model_mlp = model.evaluate(X_test2,y_test2)
#3print("\n%s: %.2f%%" % (model_,.metrics_names[1], scores[1]*100))
#y_pred= model.predict(x_test, verbose=1)

model_mlp.fit(X_trainf,y_trainf, epochs=50, batch_size=10)

