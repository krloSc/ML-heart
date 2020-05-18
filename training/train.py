from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


class training():
	best=[]
	def __init__(self):
		pass

	def search_best(self,*kwargs,heart,labels,folds=5,score='f1'):
		for svm in kwargs:
			print(svm)
			input()
			grid_search = GridSearchCV(svm.get('tipo'),
						svm.get('param'), cv=folds,
			scoring=score,verbose=1)
			grid_search.fit(heart, labels)
			self.best.append(grid_search.best_params_)
