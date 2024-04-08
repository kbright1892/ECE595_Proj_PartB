import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch import nn, cuda
import torch
from torch.optim import Adam

# check if cuda is available, if not use cpu
# cuda is not avalable on my machine, but may be on another
device = 'cuda' if cuda.is_available() else 'cpu'

class NeuralNet(nn.Module):
	def __init__(self):
		super().__init__()

		self.model = nn.Sequential(
			nn.Linear(in_features=30, out_features=20, device=device),
			nn.ReLU(),
			nn.Linear(in_features=20, out_features=20, device=device),
			nn.ReLU(),
			nn.Linear(in_features=20, out_features=2, device=device)
		)

	def forward(self, x):
		return self.model(x)


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
	train_labels = train_data['label']
	train_features = train_data.drop(columns='label')

	test_labels = test_data['label']
	test_features = test_data.drop(columns='label')

	# convert to tensors
	train_labels = torch.tensor(train_labels.to_numpy(), dtype=torch.long)
	train_features = torch.tensor(train_features.to_numpy(), dtype=torch.float32)

	test_labels = torch.tensor(test_labels.to_numpy(), dtype=torch.long)
	test_features = torch.tensor(test_features.to_numpy(), dtype=torch.float32)

	# create tensor dataset
	train_dataset = TensorDataset(train_features, train_labels)

	# create dataloaders
	train_loader = DataLoader(train_dataset, batch_size=32)

	classifier = NeuralNet()
	optimizer = Adam(classifier.parameters(), lr=0.002)
	loss_fn = nn.CrossEntropyLoss()

	for epoch in range(1000): # train for 10 epochs
		for batch in train_loader: 
			X, y = batch
			yhat = classifier(X)

			loss = loss_fn(yhat, y) 

			# Apply backprop 
			optimizer.zero_grad()
			loss.backward()
			optimizer.step() 

		print(f"Epoch {epoch + 1} loss is {loss.item()}")

	torch.save(classifier.state_dict(), 'full_model.pth')

	# test model
	
	total_correct = 0
	total = len(test_labels)

	for i in range(len(test_features)):
		if test_labels[i] == torch.argmax(classifier(test_features[i])):
			total_correct += 1
	
	print(f"Accuracy: {total_correct / total}")
	

if __name__ == '__main__':
	main()