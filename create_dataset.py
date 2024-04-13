import pandas as pd
import pickle
from sys import argv


# used to generate pickle files for the dataset to be used in the neural network
# allows for consistency and easy re-use
def main():
	# convert string to int
	num_features = int(argv[1])
	data = pd.read_csv('./dataset/wdbc.data', header=None, index_col=0, usecols=range(0, num_features + 2))

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
	column_names = ['label', 'radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension', 
				 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 
				 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst']

	data.columns = column_names[0:num_features + 1]

	# convert labels to 0 and 1
	# 1 for malignant, 0 for benign
	data['label'] = data['label'].map({'M': 1, 'B': 0})

	# split data into train and test sets using a 70/30 split
	train_data = data.sample(frac=0.7, random_state=0)
	test_data = data.drop(train_data.index)

	# split labels from features and save dataframes to pickle files for later use
	train_features = train_data.drop(columns='label')
	file = open(f'./data/{argv[1]}_features/train_features.pickle', 'wb')
	pickle.dump(train_features, file)
	file.close()	
	
	train_labels = train_data['label']
	file = open(f'./data/{argv[1]}_features/train_labels.pickle', 'wb')
	pickle.dump(train_labels, file)
	file.close()

	test_features = test_data.drop(columns='label')
	file = open(f'./data/{argv[1]}_features/test_features.pickle', 'wb')
	pickle.dump(test_features, file)
	file.close()

	test_labels = test_data['label']
	file = open(f'./data/{argv[1]}_features/test_labels.pickle', 'wb')
	pickle.dump(test_labels, file)
	file.close()


if __name__ == '__main__':
	main()