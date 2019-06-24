######################################################################################
#Data Collected From: https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection#
######################################################################################
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
import dask.dataframe as dd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score,roc_curve,recall_score,classification_report,mean_squared_error,confusion_matrix
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
import lightgbm as lgb
from contextlib import contextmanager
import time
import threading
import random
from sklearn.model_selection import train_test_split as model_tts
print(os.listdir("../data"))


# preset the data types
dtyp = {'ip': np.int64, 'app': np.int16,'device': np.int16,'os': np.int16,'channel': np.int16,'is_attributed' : np.int16}


print("LOADING DATA..........................")
# TRAINING DATA
print("TRAINING DATA")

dfTrain = pd.read_csv("../data/train.csv", nrows=100000)

#need to skip 0th row as it is the header
dfTest = pd.read_csv("../data/train.csv", skiprows=range(1, 100000), nrows=100000)

print("Loading Completed")



check_true_positive_ratio(dfTest)
dfTest = cleaning_transforming(dfTest)



print("original dataframe")
dfTrain.head()


dfTest.head()


def cleaning_transforming(dataframe):
    #this column is completely blank
    del dataframe['attributed_time']
    # Create new features out of time. Year and month are skipped as the data is only for 4 days
    dataframe['click_time'] =  dd.to_datetime(dataframe['click_time'])

    # the given data is of 4 days. So useful data is day and hours
    dataframe['day'] = dataframe['click_time'].dt.day
    dataframe['hour'] = dataframe['click_time'].dt.hour
    del dataframe['click_time']

    dataframe.columns = ['ip', 'app', 'device', 'os','channel','is_attributed','day','hour']

    print("dataset columns",dataframe.columns)

    dataframe.astype(dtyp)
    print("\n\n=============================================================")
    print(dataframe.info())
    return dataframe

def check_true_positive_ratio(data):
    nrows = len(data)
    print("Number of rows in the dataframe: ", nrows)
    npositive = data.is_attributed.sum() #since is_attributed has either 0 or 1. 1 is for positive cases
    print("Number of positive cases are " + str(npositive))
    nnegative = nrows - npositive
    positive_ratio = np.longdouble(npositive/nrows)
    print("Positive data ratio is ", positive_ratio*100, "%")




def positive_set(dataframe):
    #nRows = len(dataframe)
    #nPos = dataframe.is_attributed.sum()
    #nNeg = nRows - nPos
    #r = np.longdouble(nPos/nRows)
    posEx = dataframe [ (dataframe['is_attributed'] == 1) ]
    #print("Number of positive data is " + str(len(posEx)) + " rows")
    return posEx


def downsampling(dataframe, posDataframe):
    nRows = len(dataframe)
    nPos = len(posDataframe)
    nNeg = nRows - nPos
    r = np.longdouble(nPos/nRows)
    #randomly shuffle the dataframe and pick negative rows with shuffling and replacement
    random_int = random.randint(1,50)
    random_state = np.random.RandomState(random_int)
    sampledNegEx =  dataframe [ (dataframe['is_attributed'] == 0) ].sample(frac=r,random_state=random_state)
    newTrainsubs = [posDataframe, sampledNegEx]
    dfTrainBal = pd.concat(newTrainsubs)
    newRows = len(dfTrainBal)
    newPos = dfTrainBal.is_attributed.sum()
    newNeg = nRows - nPos
    rr = np.longdouble(newPos/newRows)
    return dfTrainBal


def create_chunks(dataframe):
    print("Creating 10 chunks")
    #get only positive data
    pos_set = positive_set(dataframe)
    #downsample the data to get balanced chunk
    balanced_chunks = []
    for i in range(10):
        balanced_chunks.append(downsampling(dataframe, pos_set))
    
    return balanced_chunks
    

def draw_roc(clf, chunk):
    #building data for crossvalidation
    random_int = random.randint(1,10)
    random_state = np.random.RandomState(random_int)
    split_size = 0.3
    dTrain, dCV = model_tts(chunk, test_size=split_size, random_state=random_state, shuffle=True )
    #Get X and y
    yTrain = dTrain['is_attributed']
    xTrain = dTrain.drop('is_attributed',axis=1)
    yCV = dCV['is_attributed']
    xCV = dCV.drop('is_attributed',axis=1)
    tprs = []
    aucs = []
    result_dict = {}
    
    mean_fpr = np.linspace(0, 1, 100)
    model = clf.fit(xTrain, yTrain)
    probas_ = model.predict_proba(xCV)
    fpr, tpr, thresholds = roc_curve(yCV, probas_[:, 1])
    
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (0, roc_auc))
    result_dict["model"] = model
    result_dict["fpr"] = fpr
    result_dict["tpr"] = tpr
    result_dict["lw"] = 1
    result_dict["alpha"] = 0.3
    result_dict["roc_fold"] = 0
    result_dict["roc_auc"] = roc_auc
    #print(str(clf), str(roc_auc))
    return result_dict



def get_roc_test(model, test_data):
    yTest = test_data['is_attributed']
    xTest = test_data.drop('is_attributed',axis=1)
    tprs = []
    aucs = []
    result_dict = {}
    mean_fpr = np.linspace(0, 1, 100)
    probas_ = model.predict_proba(xTest)
    fpr, tpr, thresholds = roc_curve(yTest, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    #plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (0, roc_auc))
    result_dict["model"] = clf
    result_dict["fpr"] = fpr
    result_dict["tpr"] = tpr
    result_dict["lw"] = 1
    result_dict["alpha"] = 0.3
    result_dict["roc_fold"] = 0
    result_dict["roc_auc"] = roc_auc
    #print(str(clf), str(roc_auc))
    return result_dict


def display_accuracy(model, test_data):
    yTest = test_data['is_attributed']
    xTest = test_data.drop('is_attributed',axis=1)
    y_pred = model.predict(xTest)
    print("Model's on test set: {:.2f}".format(model.score(xTest, yTest)))



def display_confusion_matrix(model, test_data):
    yTest = test_data['is_attributed']
    xTest = test_data.drop('is_attributed',axis=1)
    y_pred = model.predict(xTest)
    confusion_matrx = confusion_matrix(yTest, y_pred)
    print(confusion_matrx)
    return confusion_matrx


def display_classification_report(model, test_data):
    yTest = test_data['is_attributed']
    xTest = test_data.drop('is_attributed',axis=1)
    y_pred = model.predict(xTest)
    print(classification_report(yTest, y_pred))


#Cleaning and transforming both data sets
dfTrain = cleaning_transforming(dfTrain)
dfTest = cleaning_transforming(dfTest)

#checking true positive ratio in train data
check_true_positive_ratio(dfTrain)

# #checking true positive ratio in test data
check_true_positive_ratio(dfTest)

dfTrainBal = create_chunks(dfTrain)

print("Checking positive ratio in each chunk now")
for chunk in dfTrainBal:
    check_true_positive_ratio(chunk)
    print("===============================")



dfTrainBal[0] [ (dfTrainBal[0]['is_attributed'] == 0) ].head()



dfTrainBal[1] [ (dfTrainBal[1]['is_attributed'] == 0) ].head()

dfTrainBal[2] [ (dfTrainBal[2]['is_attributed'] == 0) ].head()

dfTrainBal[3] [ (dfTrainBal[3]['is_attributed'] == 0) ].head()

#building model for logistic regression
dTrain = pd.DataFrame()

dCV = pd.DataFrame()

best_model_lr = {}

model_list_lr = []

best_auc_lr = np.float64(0)

#building model
for chunk in dfTrainBal:
    clfLR = LogisticRegression()
    results_lr = draw_roc(clfLR, chunk)
    print(results_lr["roc_auc"])
    model_list_lr.append(results_lr)


for model in model_list_lr:
    if(np.greater(model["roc_auc"], best_auc_lr)):
        best_auc_lr = model["roc_auc"]
        best_model_lr = model

print("best auc is " , best_model_lr["roc_auc"])

print("ROC curve for validation set for logistic regression")
plt.plot(best_model_lr["fpr"], 
          best_model_lr["tpr"], best_model_lr["lw"], best_model_lr["alpha"],
          label='ROC fold %d (AUC = %0.2f)' % (0, best_model_lr["roc_auc"]))

best_model_lr["model"]



#Testing best model for logistic regression
test_lr = get_roc_test(best_model_lr["model"], dfTest)
print("best model's auc on test set for logistic regression is " , test_lr["roc_auc"])
print("ROC curve for test set for logistic regression")
plt.plot(test_lr["fpr"], 
          test_lr["tpr"], test_lr["lw"], test_lr["alpha"],
          label='ROC fold %d (AUC = %0.2f)' % (0, test_lr["roc_auc"]))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()



#diplaying accuracy of logistic model
print("Logistic Model Accuracy")
display_accuracy(best_model_lr["model"], dfTest)


#printing model's confusion matrix for logistic regression
print("Logistic Model Confusion Matrix")
confsn_matrx = display_confusion_matrix(best_model_lr["model"], dfTest)
correct_predictions = confsn_matrx[0][0] + confsn_matrx[1][1]
incorrect_predictions = confsn_matrx[0][1] + confsn_matrx[1][0]
print("Number of correct predictions: ", correct_predictions)
print("Number of incorrect predictions: ", incorrect_predictions)

display_classification_report(best_model_lr["model"], dfTest)

#building model for support vector machine with linear kernel
dTrain = pd.DataFrame()

dCV = pd.DataFrame()

split_size = 0.3
best_model_svm = {}
model_list_svm = []
best_auc_svm = np.float64(0)

#building model
for chunk in dfTrainBal:
    random_int = random.randint(1,10)
    random_state = np.random.RandomState(random_int)
    #building data for crossvalidation
    dTrain, dCV = model_tts(chunk, test_size=split_size, random_state=random_state, shuffle=True )
    #Get X and y
    yTrain = dTrain['is_attributed']
    xTrain = dTrain.drop('is_attributed',axis=1)
    yCV = dCV['is_attributed']
    xCV = dCV.drop('is_attributed',axis=1)
    clfSVM = svm.SVC(kernel='linear', probability=True,random_state=random_state)
    results_svm = draw_roc(clfSVM, chunk)
    model_list_svm.append(results_svm)
    print(results_svm["roc_auc"])

for model in model_list_svm:
    if(np.greater(model["roc_auc"], best_auc_svm)):
        best_auc_lr = model["roc_auc"]
        best_model_svm = model

print("best auc is " , best_model_svm["roc_auc"])
print("ROC curve for validation set for support vector machine with linear kernel")
plt.plot(best_model_svm["fpr"], 
          best_model_svm["tpr"], best_model_svm["lw"], best_model_svm["alpha"],
          label='ROC fold %d (AUC = %0.2f)' % (0, best_model_svm["roc_auc"]))

best_model_svm["model"]

#Testing best model for svm with linear kernel
test_svm = get_roc_test(best_model_svm["model"], dfTest)
print("best model's auc on test set for svm with linear kernel is " , test_svm["roc_auc"])
print("ROC curve for test set for svm with linear kernel")
plt.plot(test_svm["fpr"], 
          test_svm["tpr"], test_svm["lw"], test_svm["alpha"],
          label='ROC fold %d (AUC = %0.2f)' % (0, test_svm["roc_auc"]))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#diplaying accuracy of svm with linear kernel
print("SVM Model Accuracy")
display_accuracy(best_model_svm["model"], dfTest)

#printing model's confusion matrix for svm
print("SVM Model Confusion Matrix")
confsn_matrx = display_confusion_matrix(best_model_svm["model"], dfTest)
correct_predictions = confsn_matrx[0][0] + confsn_matrx[1][1]
incorrect_predictions = confsn_matrx[0][1] + confsn_matrx[1][0]
print("Number of correct predictions: ", correct_predictions)
print("Number of incorrect predictions: ", incorrect_predictions)

display_classification_report(best_model_svm["model"], dfTest)

#support vector machine with gaussian kernel
dTrain = pd.DataFrame()

dCV = pd.DataFrame()

split_size = 0.3
best_model_svm_gk = {}
model_list_svm_gk = []
best_auc_svm_gk = np.float64(0)

#building model
for chunk in dfTrainBal:
    random_int = random.randint(1,10)
    random_state = np.random.RandomState(random_int)
    #building data for crossvalidation
    dTrain, dCV = model_tts(chunk, test_size=split_size, random_state=random_state, shuffle=True )
    #Get X and y
    yTrain = dTrain['is_attributed']
    xTrain = dTrain.drop('is_attributed',axis=1)
    yCV = dCV['is_attributed']
    xCV = dCV.drop('is_attributed',axis=1)
    clfSVM = svm.SVC(kernel='rbf', probability=True,random_state=random_state)
    results_svm_gk = draw_roc(clfSVM, chunk)
    model_list_svm_gk.append(results_svm_gk)
    print(results_svm_gk["roc_auc"])

for model in model_list_svm_gk:
    if(np.greater(model["roc_auc"], best_auc_svm_gk)):
        best_auc_svm_gk = model["roc_auc"]
        best_model_svm_gk = model

print("best auc is " , best_model_svm_gk["roc_auc"])
print("ROC curve for validation set for support vector machine with gaussian kernel")
plt.plot(best_model_svm_gk["fpr"], 
          best_model_svm_gk["tpr"], best_model_svm_gk["lw"], best_model_svm_gk["alpha"],
          label='ROC fold %d (AUC = %0.2f)' % (0, best_model_svm_gk["roc_auc"]))

best_model_svm_gk["model"]

#Testing best model for SVM with Gaussian Kernel
test_svm_gk = get_roc_test(best_model_svm_gk["model"], dfTest)
print("best model's auc on test set for svm with gaussian kernel is " , test_svm_gk["roc_auc"])
print("ROC curve for test set for svm with gaussian kernel")
plt.plot(test_svm_gk["fpr"], 
          test_svm_gk["tpr"], test_svm_gk["lw"], test_svm_gk["alpha"],
          label='ROC fold %d (AUC = %0.2f)' % (0, test_svm_gk["roc_auc"]))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#diplaying accuracy of svm with gaussian kernel
print("SVM Model with gaussian kernel accuracy")
display_accuracy(best_model_svm_gk["model"], dfTest)

#printing model's confusion matrix for svm with gaussian kernel
print("SVM Model with Gaussian Kernel Confusion Matrix")
confsn_matrx = display_confusion_matrix(best_model_svm_gk["model"], dfTest)
correct_predictions = confsn_matrx[0][0] + confsn_matrx[1][1]
incorrect_predictions = confsn_matrx[0][1] + confsn_matrx[1][0]
print("Number of correct predictions: ", correct_predictions)
print("Number of incorrect predictions: ", incorrect_predictions)

display_classification_report(best_model_svm_gk["model"], dfTest)

#building model with random forest
dTrain = pd.DataFrame()

dCV = pd.DataFrame()

split_size = 0.3
best_model_rf = {}
model_list_rf = []
best_auc_rf = np.float64(0)

#building model
for chunk in dfTrainBal:
    random_int = random.randint(1,10)
    random_state = np.random.RandomState(random_int)
    #building data for crossvalidation
    dTrain, dCV = model_tts(chunk, test_size=split_size, random_state=random_state, shuffle=True )
    #Get X and y
    yTrain = dTrain['is_attributed']
    xTrain = dTrain.drop('is_attributed',axis=1)
    yCV = dCV['is_attributed']
    xCV = dCV.drop('is_attributed',axis=1)
    clfRF = RandomForestClassifier(n_estimators=100,random_state=random_state)
    results_rf = draw_roc(clfRF, chunk)
    model_list_rf.append(results_rf)
    print(results_rf["roc_auc"])

for model in model_list_rf:
    if(np.greater(model["roc_auc"], best_auc_rf)):
        best_auc_rf = model["roc_auc"]
        best_model_rf = model

print("best auc is " , best_model_rf["roc_auc"])
print("ROC curve for validation set for random forest")
plt.plot(best_model_rf["fpr"], 
          best_model_rf["tpr"], best_model_rf["lw"], best_model_rf["alpha"],
          label='ROC fold %d (AUC = %0.2f)' % (0, best_model_rf["roc_auc"]))

best_model_rf["model"]

#Testing best model for Random Forest
test_rf = get_roc_test(best_model_rf["model"], dfTest)
print("best model's auc on test set for random forest is " , test_rf["roc_auc"])
# #print(dTrain.head())
print("ROC curve for test set for random forest")
plt.plot(test_rf["fpr"], 
          test_rf["tpr"], test_rf["lw"], test_rf["alpha"],
          label='ROC fold %d (AUC = %0.2f)' % (0, test_rf["roc_auc"]))
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

#diplaying accuracy of random forest
print("Random Forest Accuracy")
display_accuracy(best_model_rf["model"], dfTest)

#printing model's confusion matrix for random forest
print("Random Forest Confusion Matrix")
confsn_matrx = display_confusion_matrix(best_model_rf["model"], dfTest)
correct_predictions = confsn_matrx[0][0] + confsn_matrx[1][1]
incorrect_predictions = confsn_matrx[0][1] + confsn_matrx[1][0]
print("Number of correct predictions: ", correct_predictions)
print("Number of incorrect predictions: ", incorrect_predictions)

display_classification_report(best_model_rf["model"], dfTest)

plt.plot(test_lr["fpr"], 
          test_lr["tpr"], test_lr["lw"], test_lr["alpha"],
          label='ROC fold %d (AUC = %0.2f)' % (0, test_lr["roc_auc"]))
plt.plot(test_svm["fpr"], 
          test_svm["tpr"], test_svm["lw"], test_svm["alpha"],
          label='ROC fold %d (AUC = %0.2f)' % (0, test_svm["roc_auc"]))
plt.plot(test_rf["fpr"], 
          test_rf["tpr"], test_rf["lw"], test_rf["alpha"],
          label='ROC fold %d (AUC = %0.2f)' % (0, test_rf["roc_auc"]))

def down_sampling(dataframe):
    nRows = len(dataframe)
    nPos = dataframe.is_attributed.sum()
    nNeg = nRows - nPos
    r = np.longdouble(nPos/nRows)
    print("positive cases in training set: ", 100.0*r, "%")
    print("\nnumber of positive cases are " + str(nPos))
    # Create a balenced dataset
    posEx = dataframe [ (dataframe['is_attributed'] == 1) ]
    print("Number of positive rows " , len(posEx))
    
    #randomly shuffle the dataframe and pick negative rows with shuffling and replacement
    random_int = random.randint(1,50)
    random_state = np.random.RandomState(random_int)
    sampledNegEx =  dataframe [ (dataframe['is_attributed'] == 0) ].sample(frac=r,random_state=random_state)
    print("Number of rows selected randomly in sampled negative dataset : ",len(sampledNegEx))
    newTrainsubs = [posEx, sampledNegEx]
    dfTrainBal = pd.concat(newTrainsubs)
    print("Number of rows in balanced dataset ", len(dfTrainBal))
    #printing the new ratio pf pos neg
    newRows = len(dfTrainBal)
    newPos = dfTrainBal.is_attributed.sum()
    newNeg = nRows - nPos
    rr = np.longdouble(newPos/newRows)
    print("positive cases in training set: ", 100.0*rr, "%")
    print("\nnumber of positive cases are " + str(newPos))
    return dfTrainBal


#train_df=dfTrain
train_df = pd.read_csv("../data/train.csv", nrows=10000000)
train_df['click_time'] = pd.to_datetime(train_df['click_time'])
train_df['attributed_time'] = pd.to_datetime(train_df['attributed_time'])

train_df['hour'] = train_df.click_time.dt.hour.astype('uint8')
train_df['day'] = train_df.click_time.dt.day.astype('uint8')
train_df.drop(['attributed_time'], axis=1, inplace=True)
train_df.drop(['click_time'], axis=1, inplace=True)

train_df = down_sampling(train_df)

@contextmanager
def timer(name):
    t0=time.time()
    yield
    print(f'[{name}] done in {time.time()-t0:.0f} s')

VALIDATE = False
RANDOM_STATE = 50
VALID_SIZE = 0.30
MAX_ROUNDS = 1000
EARLY_STOP = 50
OPT_ROUNDS = 650
skiprows = range(1,134903891)
nrows = 50000000

target = 'is_attributed'
predictors = ['ip', 'app','device','os', 'channel', 'day', 'hour']
categorical = ['ip', 'app', 'device', 'os', 'channel', 'day', 'hour']

train_df, val_df = model_tts(train_df, test_size=VALID_SIZE, random_state=RANDOM_STATE, shuffle=True )

params = {
          'boosting_type': 'gbdt',
          'objective': 'binary',
          'metric':'auc',
          'learning_rate': 0.1,
          'num_leaves': 9,  # we should let it be smaller than 2^(max_depth)
          'max_depth': 5,  # -1 means no limit
          'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
          'max_bin': 100,  # Number of bucketed bin for feature values
          'subsample': 0.9,  # Subsample ratio of the training instance.
          'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
          'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
          'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
          'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
          'nthread': 8,
          'verbose': 0,
          'scale_pos_weight':200, # because training data is extremely unbalanced 
          'num_threads':32
         }
dtrain = lgb.Dataset(train_df[predictors].values, 
                             label=train_df[target].values,
                             feature_name=predictors,
                             categorical_feature=categorical)

dvalid = lgb.Dataset(val_df[predictors].values,
                             label=val_df[target].values,
                             feature_name=predictors,
                             categorical_feature=categorical)
evals_result = {}
with timer('train with valid'):
    model = lgb.train(params, dtrain, valid_sets=[dtrain, dvalid], 
                          valid_names=['train','valid'], 
                          evals_result=evals_result, 
                          num_boost_round=MAX_ROUNDS,
                          early_stopping_rounds=EARLY_STOP,
                          verbose_eval=50, 
                          feval=None)
        

print('Plotting feature importances...')
lgb.plot_importance(model, max_num_features=30)
plt.show()

test_data = dfTest
yTest = test_data['is_attributed']
xTest = test_data.drop('is_attributed',axis=1)

pred = model.predict(xTest, num_iteration=model.best_iteration)
pred


