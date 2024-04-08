import torch
import pandas as pd
import pickle


def main():
	data = pd.read_csv('./dataset/wdbc.data', header=None, index_col=0)

	'''
	columns headers for columns 2-11:
	a) radius (mean of distances from center to points on the perimeter)
	b) texture (standard deviation of gray-scale values)
	c) perimeter
	d) area
	e) smoothness (local variation in radius lengths)
	f) compactness (perimeter^2 / area - 1.0)
	g) concavity (severity of concave portions of the contour)
	h) concave points (number of concave portions of the contour)
	i) symmetry 
	j) fractal dimension ("coastline approximation" - 1)

	12-21: SE = standard error
	22-31: "worst" or largest (mean of the three largest values)
	'''

	# rename columns
	data.columns = ['label', 'radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension', 
				 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 
				 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

	# convert labels to 0 and 1
	data['label'] = data['label'].map({'M': 1, 'B': 0})

	# split data into train and test sets
	train_data = data.sample(frac=0.8, random_state=0)
	test_data = data.drop(train_data.index)

	# split labels from features
	train_features = train_data.drop(columns='label')
	file = open('./data/train_features.pickle', 'wb')
	pickle.dump(train_features, file)
	file.close()	
	
	train_labels = train_data['label']
	file = open('./data/train_labels.pickle', 'wb')
	pickle.dump(train_labels, file)
	file.close()

	test_features = test_data.drop(columns='label')
	file = open('./data/test_features.pickle', 'wb')
	pickle.dump(test_features, file)
	file.close()

	test_labels = test_data['label']
	file = open('./data/test_labels.pickle', 'wb')
	pickle.dump(test_labels, file)
	file.close()


if __name__ == '__main__':
	main()