import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch
from torch.optim import Adam
from Model import NeuralNet


def main():
	train_features = pd.read_pickle('./data/train_features.pickle')
	train_labels = pd.read_pickle('./data/train_labels.pickle')

	# convert to tensors
	train_labels = torch.tensor(train_labels.to_numpy(), dtype=torch.long)
	train_features = torch.tensor(train_features.to_numpy(), dtype=torch.float32)

	# create tensor dataset
	train_dataset = TensorDataset(train_features, train_labels)

	# create dataloaders
	train_loader = DataLoader(train_dataset, batch_size=32)

	# create model, optimizer, and loss function
	classifier = NeuralNet()
	optimizer = Adam(classifier.parameters(), lr=0.002)
	loss_fn = nn.CrossEntropyLoss()

	# train model
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

	# save model
	torch.save(classifier.state_dict(), 'full_model.pth')
	
	
	

if __name__ == '__main__':
	main()