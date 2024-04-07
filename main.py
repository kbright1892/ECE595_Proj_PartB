import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import cuda, nn, LongTensor, FloatTensor, tensor
import torch
from torch.optim import Adam

# check if cuda is available, if not use cpu
# cuda is not avalable on my machine, but may be on another
device = 'cuda' if cuda.is_available() else 'cpu'

class NeuralNet(nn.Module):
	def __init__(self):
		super().__init__()

		self.model = nn.Sequential(
			nn.Linear(in_features=8, out_features=8, device=device),
			nn.ReLU(),
			nn.Linear(in_features=8, out_features=2, device=device),
			nn.ReLU(),
		)

		self.double()

	def forward(self, x):
		return self.model(x)
	
	
class MyDataset(Dataset):
	def __init__(self, data: pd.DataFrame):
		self.data = data

	def __getitem__(self, index):
		row = self.data.iloc[index].to_numpy()
		features = row[1:]
		label = row[0]
		return features, label

	def __len__(self):
		return len(self.data)


def main():
	data = pd.read_csv('./dataset/wdbc.data', header=None, index_col=0)
	
	# remove columns 12-31, which are not real value features
	data = data.drop(columns=range(12, 32))

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
	'''

	# rename columns
	data.columns = ['label', 'radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness', 'concavity', 'concave_points', 'symmetry', 'fractal_dimension']

	# convert labels to 0 and 1
	data['label'] = data['label'].map({'M': 1, 'B': 0})

	# split data into train and test sets
	train_data = data.sample(frac=0.8, random_state=0)

	test_data = data.drop(train_data.index)

	# create dataset
	train_data = MyDataset(data=train_data)

	# create dataloaders
	train_loader = DataLoader(train_data, batch_size=32)
	test_loader = DataLoader(test_data, batch_size=32)

	classifier = NeuralNet()
	optimizer = Adam(classifier.parameters(), lr=0.001)
	loss_fn = nn.CrossEntropyLoss()

	for epoch in range(10): # train for 10 epochs
		for batch in train_loader: 
			X, y = batch
			yhat = classifier(X)

			yhat = yhat.type(LongTensor)

			loss = loss_fn(yhat, y) 

			# Apply backprop 
			optimizer.zero_grad()
			loss.backward()
			optimizer.step() 

		print(f"Epoch:{epoch} loss is {loss.item()}")
	
	

if __name__ == '__main__':
	main()