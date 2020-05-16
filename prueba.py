from util.data import *
####################   preprocessing Data   ###################################
data=Data()
data.import_data("heart")
data.correlation("result")
#data.corr_matrix(save=True)
data.encoder("cp")
train_set, test_set = data.split(0.2)
train_labels=train_set["result"].copy()
train_labels=test_set["result"].copy()
train_prepared, test_prepared = data.prepare()
x,y=data.split(0.2)
print(x.shape,y.shape)
###########################    Training    ####################################
