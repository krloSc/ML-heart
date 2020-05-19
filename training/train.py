from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


class training():
	best_params=[]
	best_models=[]
	def __init__(self):
		pass

	def search_best(self,*kwargs,heart,labels,folds=5,score='f1',out=False):
		for svm in kwargs:
			grid_search = GridSearchCV(svm.get('tipo'),
						svm.get('param'), cv=folds,
			scoring=score,verbose=out)
			grid_search.fit(heart, labels)
			self.best_params.append(grid_search.best_params_)
			self.best_models.append(grid_search.best_estimator_)
