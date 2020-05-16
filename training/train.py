from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
best=[]

def div(data,size , random=42):
	train_set, test_set=train_test_split(data, test_size=size, random_state=random)
	return train_set, test_set 
	
def process(heart,enc="cp",encode=False):
	heart_num=heart.drop([enc],axis=1)
	num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
	])
	num_attribs = list(heart_num)
	cat_attribs = [enc]

	full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
	])
	if encode==True:
		return full_pipeline.fit_transform(heart)
	else:
		return num_pipeline.fit_transform(heart)

def search_best(tipo,heart_prepared,heart_labels,param,folds=5,score='f1'):
	param_grid = [param]
	grid_search = GridSearchCV(tipo, param_grid, cv=folds,
	scoring=score,verbose=1)
	grid_search.fit(heart_prepared, heart_labels)
	global best
	best.append(grid_search.best_params_)
	return grid_search.best_estimator_
