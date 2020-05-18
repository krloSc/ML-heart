from util.data import *
from training.train import*
from sklearn.metrics import precision_score, recall_score, confusion_matrix

####################   preprocessing Data   ###################################

data=Data()
data.import_data("heart")
data.correlation("result")
#data.corr_matrix(save=True)
data.encoder("cp")
train_set, test_set = data.split(0.2)
train_labels=train_set["result"].copy()
test_labels=test_set["result"].copy()
train_prepared, test_prepared = data.prepare()

###########################    Training    ####################################

train=training()
params=({'tipo':SVC(),'param':{'kernel': ["rbf"], 'coef0': [0, 1,10], 'gamma': [ 0.1, 1, 5], 'C':[ 0.01, 1, 5]}},
    {'tipo':SVC(),'param':{'kernel': ["poly"],'degree':[2,3], 'coef0': [3,5,10,50], 'C':[ 0.01, 1, 5,30]}},
    {'tipo':LinearSVC(),'param':{'C':[0.1,1,3,5],'max_iter':[10000]}},
    {'tipo':LinearSVC(),'param':{'max_iter':[100000],'C':[0.1,1,3,5]}})
training.search_best(training,*params,heart=train_prepared,labels=train_labels)


##############################   Evaluating   ###################################

for clf, title in zip(training.best_models,training.best_params):
    clf.fit(train_prepared,train_labels)
    print(str(title),"\t",recall_score(clf.predict(test_prepared),test_labels),"\t",
        precision_score(clf.predict(test_prepared),test_labels))
