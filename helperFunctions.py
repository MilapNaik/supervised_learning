import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from sklearn.tree import export_graphviz

def read_cancer_csv(file):
	data = pd.read_csv(file)

	# If breast cancer:
	# 11. Class: (2 for benign, 4 for malignant)
	y = data.iloc[:,[10]]
	x = data.iloc[:,:10]

	# preprocess the ? to median values - only in x["Bare Nuclei"]
	stack = x.stack()
	stack[ stack == '?' ] = x["Bare Nuclei"][ x["Bare Nuclei"] != '?' ].median()
	x = stack.unstack()

	# imputer = SimpleImputer(missing_values=np.nan, strategy="median")
	# imputer.fit(x["Bare Nuclei"])
	# x["Bare Nuclei"] = imputer.transform(x["Bare Nuclei"])

	median = x
	print(x)
	# print(x["Bare Nuclei"])
	print(y)

	return x, y

def read_fertility_csv(file):
	data = pd.read_csv(file)

	# If breast cancer:
	# 11. Class: (2 for benign, 4 for malignant)
	y = data.iloc[:,[9]]
	x = data.iloc[:,:9]

	# preprocess the ? to median values - only in x["Bare Nuclei"]
	# stack = x.stack()
	# stack[ stack == '' ] = x["Bare Nuclei"][ x["Bare Nuclei"] != '' ].median()
	# x = stack.unstack()

	# imputer = SimpleImputer(missing_values=np.nan, strategy="median")
	# imputer.fit(x)
	# x = imputer.transform(x)

	# median = x
	print(x)
	# print(x["Bare Nuclei"])
	print(y)

	return x, y

	# Vizualize
def vizualize_cancer_tree(tree):
	export_graphviz(
		tree,
		out_file="images/cancer_tree.dot",
		feature_names=['ID Number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
									'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei',
									'Bland Chromatin', 'Normal Nucleoli', 'Mitoses'],
		class_names='Class',
		rounded=True,
		filled=True
	)

def vizualize_fertility_tree(tree):
	export_graphviz(
		tree,
		out_file="images/fertility_tree.dot",
		feature_names=['Season', 'Age', 'Childhood diseases', 'Accident or serious trauma', 'Surgical intervention', 'High fevers in the last year', 'Frequency of alcohol consumption', 'Smoking', 'Number of hours spent sitting per day'],
		class_names='Class',
		rounded=True,
		filled=True
	)