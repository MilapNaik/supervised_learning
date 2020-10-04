from helperFunctions import read_cancer_csv, read_fertility_csv, vizualize_cancer_tree, vizualize_fertility_tree

import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve, validation_curve, GridSearchCV

import matplotlib.pyplot as plt

breast_cancer_data = "data/breast-cancer-wisconsin-data.csv"
breast_cancer = False
fertility_data = "data/fertility_Diagnosis.csv"
fertility = not breast_cancer

seed = 50
dtree = False
nn = False
boosting = False
svm = True
knn = False

if breast_cancer:
	x, y = read_cancer_csv(breast_cancer_data)
else:
	x, y = read_fertility_csv(fertility_data)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=seed)

# Decision tree
if dtree:
	# Build tree and test accuracy
	tree = DecisionTreeClassifier(max_depth=6, min_samples_split=4, random_state=seed)
	tree.fit(x_train,y_train)
	y_predicted = tree.predict(x_test)
	tree_accuracy = accuracy_score(y_test, y_predicted)
	print("tree accuracy:", tree_accuracy*100)

	# Grid search for best hyperparams
	param_1 = np.arange(1, 10)
	accuracies = []
	for param in param_1:
		tree = DecisionTreeClassifier(max_depth=param, min_samples_split=4, random_state=seed)
		tree.fit(x_train,y_train)
		y_predicted = tree.predict(x_test)
		tree_accuracy = accuracy_score(y_test, y_predicted)
		print("tree accuracy:", tree_accuracy*100)
		accuracies.append(tree_accuracy)

	plt.plot(param_1, accuracies, label='Accuracy')
	plt.xlabel('Max Depth')
	plt.ylabel("Score")
	plt.legend(loc="best")
	plt.grid()
	plt.title('HP Tuning - Max Depth')
	plt.savefig('images/dtree_HP_tuning_2.png')

	param_2 = np.arange(2, 12)
	parameters = {
		'criterion':['gini', 'entropy'],
		'splitter':['best', 'random'],
		'max_depth': param_1,
		'min_samples_split': param_2
	}

	clf = GridSearchCV(DecisionTreeClassifier(random_state=seed), param_grid=parameters, cv=5)
	clf.fit(x_train, y_train)
	print("best dree estimator:", clf.best_estimator_)
	print("best dtree params:", clf.best_params_)
	y_predicted = clf.predict(x_test)
	tree_accuracy = accuracy_score(y_test, y_predicted)
	print("tree accuracy after params:", tree_accuracy*100)

	# Vizualize
	if breast_cancer:
		# Tree with best HP
		tree = DecisionTreeClassifier(max_depth=4, min_samples_split=2, splitter='random', random_state=seed)
		tree.fit(x_train,y_train)
		y_predicted = tree.predict(x_test)
		tree_accuracy = accuracy_score(y_test, y_predicted)
		print("tree accuracy:", tree_accuracy*100)

		vizualize_cancer_tree(tree)

		# Learning Curve
		train_sizes = np.linspace(0.01, 1.0, 10)
		train_sizes_abs, train_scores, test_scores = learning_curve(tree, x_train, y_train, train_sizes=train_sizes)

		plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
		plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Testing score')
		plt.xlabel('Training Set Size (%)')
		plt.ylabel("Score")
		plt.legend(loc="best")
		plt.grid()
		plt.title('Decision Tree Learning Curve')
		plt.savefig('images/dtree_learning_curve.png')

		# Model Complexity - (validation curve)
		val_train_scores, val_test_scores = validation_curve(tree, x_train, y_train, param_name="max_depth", param_range=param_1)

		plt.figure()
		plt.plot(param_1, np.mean(val_train_scores, axis=1), label='Training score')
		plt.plot(param_1, np.mean(val_test_scores, axis=1), label='Testing score')
		plt.xlabel('Max depth')
		plt.ylabel("Score")
		plt.legend(loc="best")
		plt.grid()
		plt.title('Decision Tree Validation Curve')
		plt.savefig('images/dtree_validation_curve_1.png')

		val_train_scores, val_test_scores = validation_curve(tree, x_train, y_train, param_name="min_samples_split", param_range=param_2)

		plt.figure()
		plt.plot(param_2, np.mean(val_train_scores, axis=1), label='Training score')
		plt.plot(param_2, np.mean(val_test_scores, axis=1), label='Testing score')
		plt.xlabel('min_samples_split')
		plt.ylabel("Score")
		plt.legend(loc="best")
		plt.grid()
		plt.title('Decision Tree Validation Curve')
		plt.savefig('images/dtree_validation_curve_2.png')
	elif fertility:
		# Tree with best HP
		tree = DecisionTreeClassifier(max_depth=2, min_samples_split=2, splitter='random', random_state=seed)
		tree.fit(x_train,y_train)
		y_predicted = tree.predict(x_test)
		tree_accuracy = accuracy_score(y_test, y_predicted)
		print("fertility tree accuracy:", tree_accuracy*100)

		vizualize_fertility_tree(tree)

		train_sizes = np.linspace(0.01, 1.0, 10)
		train_sizes_abs, train_scores, test_scores = learning_curve(tree, x_train, y_train, train_sizes=train_sizes)

		plt.figure()
		plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
		plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Testing score')
		plt.xlabel('Training Set Size (%)')
		plt.ylabel("Score")
		plt.legend(loc="best")
		plt.grid()
		plt.title('Decision Tree Learning Curve')
		plt.savefig('images/dtree_learning_curve_fertility.png')

		# Model Complexity - (validation curve)
		val_train_scores, val_test_scores = validation_curve(tree, x_train, y_train, param_name="max_depth", param_range=param_1)

		plt.figure()
		plt.plot(param_1, np.mean(val_train_scores, axis=1), label='Training score')
		plt.plot(param_1, np.mean(val_test_scores, axis=1), label='Testing score')
		plt.xlabel('Max depth')
		plt.ylabel("Score")
		plt.legend(loc="best")
		plt.grid()
		plt.title('Decision Tree Validation Curve')
		plt.savefig('images/dtree_validation_curve_1_fertility.png')

		val_train_scores, val_test_scores = validation_curve(tree, x_train, y_train, param_name="min_samples_split", param_range=param_2)

		plt.figure()
		plt.plot(param_2, np.mean(val_train_scores, axis=1), label='Training score')
		plt.plot(param_2, np.mean(val_test_scores, axis=1), label='Testing score')
		plt.xlabel('min_samples_split')
		plt.ylabel("Score")
		plt.legend(loc="best")
		plt.grid()
		plt.title('Decision Tree Validation Curve')
		plt.savefig('images/dtree_validation_curve_2_fertility.png')


# Neural Network
if nn:
	print("starting Artificial Neural Network")
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=seed)
	clf.fit(x_train, y_train.values.ravel())
	y_predicted = clf.predict(x_test)
	nn_accuracy = accuracy_score(y_test, y_predicted)
	print("NN accuracy:", nn_accuracy*100)

	parameters = {'solver': ['lbfgs'], 'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ], 'alpha': 10.0 ** -np.arange(1, 10), 'hidden_layer_sizes':np.arange(10, 15), 'random_state':[0,1,2,3,4,5,6,7,8,9]}
	clf = GridSearchCV(MLPClassifier(), parameters, verbose= 1, n_jobs=-1)
	clf.fit(x_train, y_train.values.ravel())
	print(clf.score(x_train, y_train))
	# print("best NN estimator:", clf.best_estimator_)
	print("best NN params:", clf.best_params_)

	if breast_cancer:
		clf = MLPClassifier(alpha=0.1, batch_size='auto', early_stopping=False,
              hidden_layer_sizes=3, max_iter=1000, random_state=seed)
		clf.fit(x_train, y_train.values.ravel())
		y_predicted = clf.predict(x_test)
		nn_accuracy = accuracy_score(y_test, y_predicted)
		print("best NN accuracy:", nn_accuracy*100)

		train_sizes = np.linspace(0.01, 1.0, 10)
		train_sizes_abs, train_scores, test_scores = learning_curve(clf, x_train, y_train.values.ravel(), train_sizes=train_sizes)

		plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
		plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Testing score')
		plt.xlabel('Training Set Size (%)')
		plt.ylabel("Score")
		plt.legend(loc="best")
		plt.grid()
		plt.title('Neural Network Learning Curve')
		plt.savefig('images/nn_learning_curve.png')

		# Model Complexity - (validation curve)
		# param_1 = [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ]
		param_1 = np.arange(1, 15)
		val_train_scores, val_test_scores = validation_curve(clf, x_train, y_train.values.ravel(), param_name="hidden_layer_sizes", param_range=param_1)

		plt.figure()
		plt.plot(param_1, np.mean(val_train_scores, axis=1), label='Training score')
		plt.plot(param_1, np.mean(val_test_scores, axis=1), label='Testing score')
		plt.xlabel('Hidden Layer Sizes')
		plt.ylabel("Score")
		plt.legend(loc="best")
		plt.grid()
		plt.title('Neural Network Validation Curve')
		plt.savefig('images/NN_validation_curve_1.png')
	elif fertility:
		clf = MLPClassifier(alpha=0.1, batch_size='auto', early_stopping=False,
              hidden_layer_sizes=3, max_iter=10000, random_state=seed)
		clf.fit(x_train, y_train.values.ravel())
		y_predicted = clf.predict(x_test)
		nn_accuracy = accuracy_score(y_test, y_predicted)
		print("best NN accuracy:", nn_accuracy*100)

		train_sizes = np.linspace(0.1, 1.0, 9)
		train_sizes_abs, train_scores, test_scores = learning_curve(clf, x_train, y_train.values.ravel(), train_sizes=train_sizes)

		plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
		plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Testing score')
		plt.xlabel('Training Set Size (%)')
		plt.ylabel("Score")
		plt.legend(loc="best")
		plt.grid()
		plt.title('Neural Network Learning Curve')
		plt.savefig('images/nn_learning_curve_2.png')

		# Model Complexity - (validation curve)
		# param_1 = [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ]
		param_1 = np.arange(1, 15)
		val_train_scores, val_test_scores = validation_curve(clf, x_train, y_train.values.ravel(), param_name="hidden_layer_sizes", param_range=param_1)

		plt.figure()
		plt.plot(param_1, np.mean(val_train_scores, axis=1), label='Training score')
		plt.plot(param_1, np.mean(val_test_scores, axis=1), label='Testing score')
		plt.xlabel('Hidden Layer Sizes')
		plt.ylabel("Score")
		plt.legend(loc="best")
		plt.grid()
		plt.title('Neural Network Validation Curve')
		plt.savefig('images/NN_validation_curve_2.png')

# Boosting
if boosting:
	print("starting Boosting")
	if breast_cancer:
		tree = DecisionTreeClassifier(max_depth=4, min_samples_split=2, splitter='random', random_state=seed)
		accuracies = []
		lr = np.linspace(0.1, 1.0, 9)
		for i in lr:
			print(i)
			ada_clf = AdaBoostClassifier(tree, n_estimators=200, algorithm="SAMME.R", learning_rate=i, random_state=seed)
			ada_clf.fit(x_train, y_train.values.ravel())
			y_predicted = ada_clf.predict(x_test)
			boosting_accuracy = accuracy_score(y_test, y_predicted)
			print("Boosting accuracy:", boosting_accuracy*100)
			accuracies.append(boosting_accuracy)

		plt.plot(lr, accuracies, label='Accuracy')
		plt.xlabel('Learning Rate')
		plt.ylabel("Score")
		plt.legend(loc="best")
		plt.grid()
		plt.title('HP Tuning - learning rate')
		plt.savefig('images/boosting_learning_rate_tuning.png')

		n_ests = np.arange(1, 400)
		learning = np.linspace(0.01, 1.0, 10)
		parameters = {
			'n_estimators': n_ests,
			'base_estimator__criterion' : ["gini", "entropy"],
			'base_estimator__splitter' :   ["best", "random"],
			'learning_rate': learning
		}
		clf = GridSearchCV(AdaBoostClassifier(base_estimator = tree), parameters, verbose= 1, n_jobs=-1)
		clf.fit(x_train, y_train.values.ravel())
		print(clf.score(x_train, y_train))
		# print("best NN estimator:", clf.best_estimator_)
		print("best boosting params:", clf.best_params_)

		train_sizes = np.linspace(0.01, 1.0, 10)
		train_sizes_abs, train_scores, test_scores = learning_curve(ada_clf, x_train, y_train.values.ravel(), train_sizes=train_sizes)

		plt.figure()
		plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
		plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Testing score')
		plt.xlabel('Training Set Size (%)')
		plt.ylabel("Score")
		plt.legend(loc="best")
		plt.grid()
		plt.title('Boosting Learning Curve')
		plt.savefig('images/boosting_learning_curve.png')
	elif fertility:
		tree = DecisionTreeClassifier(max_depth=4, min_samples_split=2, splitter='random', random_state=seed)
		accuracies = []
		lr = np.linspace(0.1, 1.0, 9)
		for i in lr:
			print(i)
			ada_clf = AdaBoostClassifier(tree, n_estimators=200, algorithm="SAMME.R", learning_rate=i, random_state=seed)
			ada_clf.fit(x_train, y_train.values.ravel())
			y_predicted = ada_clf.predict(x_test)
			boosting_accuracy = accuracy_score(y_test, y_predicted)
			print("Boosting accuracy:", boosting_accuracy*100)
			accuracies.append(boosting_accuracy)

		plt.plot(lr, accuracies, label='Accuracy')
		plt.xlabel('Learning Rate')
		plt.ylabel("Score")
		plt.legend(loc="best")
		plt.grid()
		plt.title('HP Tuning - learning rate')
		plt.savefig('images/boosting_learning_rate_tuning_2.png')

		train_sizes = np.linspace(0.2, 1.0, 8)
		train_sizes_abs, train_scores, test_scores = learning_curve(ada_clf, x_train, y_train.values.ravel(), train_sizes=train_sizes)

		plt.figure()
		plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
		plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Testing score')
		plt.xlabel('Training Set Size (%)')
		plt.ylabel("Score")
		plt.legend(loc="best")
		plt.grid()
		plt.title('Boosting Learning Curve')
		plt.savefig('images/boosting_learning_curve_2.png')

# Support Vector Machines
if svm:
	print("starting SVM")
	# svm_clf = Pipeline([
	# 			("scalar", StandardScaler()),
	# 			("linear_svc", LinearSVC(C=1, loss="hinge")),
	# 		])
	svm_clf = Pipeline([('scale', StandardScaler()),
                   ('SVC',LinearSVC(dual=False))])
	svm_clf.fit(x_train, y_train.values.ravel())
	y_predicted = svm_clf.predict(x_test)
	svm_accuracy = accuracy_score(y_test, y_predicted)
	print("SVM accuracy:", svm_accuracy*100)

	scores = ['precision', 'recall']
	accuracy = []
	for score in scores:
		print("# Tuning hyper-parameters for %s" % score)
		print()
		param_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
						'C': [1, 10, 100, 1000]},
						{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
		linearSVC = GridSearchCV(
			SVC(), param_grid, scoring='%s_macro' % score
		)
		linearSVC.fit(x_train, y_train.values.ravel())
		print(linearSVC.score(x_train, y_train))
		# print("best NN estimator:", clf.best_estimator_)
		y_predicted = linearSVC.predict(x_test)
		svm_accuracy = accuracy_score(y_test, y_predicted)
		print("SVM accuracy:", svm_accuracy*100)
		accuracy.append(svm_accuracy)
		print("best svm params:", linearSVC.best_params_)

	plt.plot(scores, accuracy, label='Accuracy')
	plt.xlabel('Scoring')
	plt.ylabel("Accuracy")
	plt.legend(loc="best")
	plt.grid()
	plt.title('HP Tuning - Scoring')
	plt.savefig('images/SVM_neighbors_tuning_2.png')


	if breast_cancer:
		svm_clf = Pipeline([
				("scalar", StandardScaler()),
				("linear_svc", LinearSVC(C=0.01, loss="hinge")),
			])
		svm_clf.fit(x_train, y_train.values.ravel())
		y_predicted = svm_clf.predict(x_test)
		svm_accuracy = accuracy_score(y_test, y_predicted)
		print("best SVM accuracy:", svm_accuracy*100)

		train_sizes = np.linspace(0.1, 1.0, 9)
		train_sizes_abs, train_scores, test_scores = learning_curve(svm_clf, x_train, y_train.values.ravel(), train_sizes=train_sizes)

		plt.figure()
		plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
		plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Testing score')
		plt.xlabel('Training Set Size (%)')
		plt.ylabel("Score")
		plt.legend(loc="best")
		plt.grid()
		plt.title('SVM Learning Curve')
		plt.savefig('images/svm_learning_curve.png')

	elif fertility:
		svm_clf = Pipeline([
				("scalar", StandardScaler()),
				("linear_svc", LinearSVC(C=0.01, loss="hinge")),
			])
		svm_clf.fit(x_train, y_train.values.ravel())
		y_predicted = svm_clf.predict(x_test)
		svm_accuracy = accuracy_score(y_test, y_predicted)
		print("best SVM accuracy:", svm_accuracy*100)

		train_sizes = np.linspace(0.1, 1.0, 9)
		train_sizes_abs, train_scores, test_scores = learning_curve(svm_clf, x_train, y_train.values.ravel(), train_sizes=train_sizes)

		plt.figure()
		plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
		plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Testing score')
		plt.xlabel('Training Set Size (%)')
		plt.ylabel("Score")
		plt.legend(loc="best")
		plt.grid()
		plt.title('SVM Learning Curve')
		plt.savefig('images/svm_learning_curve_2.png')

# K Nearest Neighbors
if knn:
	print("starting KNN")
	neighbors = [1, 3, 5, 11, 15, 19, 23]
	accuracies = []
	for neighbor in neighbors:
		knn = KNeighborsClassifier(n_neighbors=neighbor)
		knn.fit(x_train, y_train.values.ravel())
		y_predicted = knn.predict(x_test)
		knn_accuracy = accuracy_score(y_test, y_predicted)
		print("KNN accuracy with", neighbor, "neighbors:", knn_accuracy*100)
		accuracies.append(knn_accuracy)

	plt.plot(neighbors, accuracies, label='Accuracy')
	plt.xlabel('Number of Neighbors')
	plt.ylabel("Score")
	plt.legend(loc="best")
	plt.grid()
	plt.title('HP Tuning - N Neighbors')
	plt.savefig('images/knn_neighbors_tuning_2.png')

	grid_params = {
		'n_neighbors': neighbors,
		'weights': ['uniform', 'distance'],
		'metric':['euclidean', 'manhattan']
	}
	gs = GridSearchCV(
		KNeighborsClassifier(),
		grid_params,
		verbose= 1,
		cv= 3,
		n_jobs= -1
	)

	results = gs.fit(x_train, y_train.values.ravel())

	print("best KNN score:",results.best_score_)
	print("best KNN estimator:",results.best_estimator_)
	print("best KNN params:",results.best_params_)

	if breast_cancer:
		train_sizes = np.linspace(0.13, 1.0, 10)
		train_sizes_abs, train_scores, test_scores = learning_curve(knn, x_train, y_train.values.ravel(), train_sizes=train_sizes)

		plt.figure()
		plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
		plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Testing score')
		plt.xlabel('Training Set Size (%)')
		plt.ylabel("Score")
		plt.legend(loc="best")
		plt.grid()
		plt.title('KNN Learning Curve')
		plt.savefig('images/knn_learning_curve.png')

	elif fertility:
		train_sizes = np.linspace(0.6, 1.0, 10)
		train_sizes_abs, train_scores, test_scores = learning_curve(knn, x_train, y_train.values.ravel(), train_sizes=train_sizes)

		plt.figure()
		plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
		plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Testing score')
		plt.xlabel('Training Set Size (%)')
		plt.ylabel("Score")
		plt.legend(loc="best")
		plt.grid()
		plt.title('KNN Learning Curve')
		plt.savefig('images/knn_learning_curve_2.png')


