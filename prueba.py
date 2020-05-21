from util.data import *
from util import image
from training.train import*
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.decomposition import PCA
####################   preprocessing Data   ###################################

data=Data()
data.import_data("heart")
data.correlation("result")
data.corr_matrix(save=True)
data.encoder('cp')
train_set, test_set = data.split(0.3)
train_labels=train_set["result"].copy()
test_labels=test_set["result"].copy()
train_prepared, test_prepared = data.prepare()

###########################    Training    ####################################
pca=PCA(n_components=2)
train_prepared=pca.fit_transform(train_prepared)
test_prepared=pca.fit_transform(test_prepared)
train=training()
params=({'tipo':SVC(),'param':{'kernel': ["rbf"], 'coef0': [0, 1,10], 'gamma': [ 0.1, 1, 5], 'C':[ 0.01, 1, 5]}},
    {'tipo':SVC(),'param':{'kernel': ["poly"],'degree':[2,3], 'coef0': [3,5,10,50], 'C':[ 0.01, 1, 5,30]}},
    {'tipo':LinearSVC(),'param':{'C':[0.1,1,3,5],'max_iter':[10000]}},
    {'tipo':LinearSVC(),'param':{'max_iter':[100000],'C':[0.1,1,3,5]}})
training.search_best(training,*params,heart=train_prepared,labels=train_labels,out=False)

###############################################################################

print('################################   Evaluating   #####################################')
print('{0:70} Recall Precision'.format('Parameters'))
for clf, title in zip(training.best_models,training.best_params):
    clf.fit(train_prepared,train_labels)
    print('{0:70}  %.3f %.3f'.format(str(title))
        %(recall_score(clf.predict(test_prepared),test_labels),
        precision_score(clf.predict(test_prepared),test_labels)))
image.visual(training.best_models,training.best_params,train_prepared,train_labels)
input("Press enter key to finish")
