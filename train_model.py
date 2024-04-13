import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torch
from torch.optim import Adam
from NeuralNet import NeuralNet
from sys import argv
from evaluate_model import main as evaluate_model


# performs initial training on the model using the training features and labels
# results are saved in a csv file called train_{number of features}_features.csv
# requires the number of features as a command line argument
def main():
	# load training features and labels
	train_features = pd.read_pickle(f'./data/{argv[1]}_features/train_features.pickle')
	train_labels = pd.read_pickle(f'./data/{argv[1]}_features/train_labels.pickle')

	# convert to tensors
	train_labels = torch.tensor(train_labels.to_numpy(), dtype=torch.long)
	train_features = torch.tensor(train_features.to_numpy(), dtype=torch.float32)

	# create tensor dataset
	train_dataset = TensorDataset(train_features, train_labels)

	# create dataloader
	train_loader = DataLoader(train_dataset, batch_size=32)

	# initial learning rate
	learning_rate = 0.001

	# open file to save results
	f = open(f'results/train_{argv[1]}_features.csv', 'w')
	# write headers
	f.write('Learning Rate,Train Accuracy,Test Accuracy\n')

	# train model with 5 different learning rates for 1000 epochs using the Adam optimizer and CrossEntropyLoss
	for i in range(0, 5):
		# create model, optimizer, and loss function
		classifier = NeuralNet(int(argv[1]))
		optimizer = Adam(classifier.parameters(), lr=learning_rate)
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

		# evaluate model
		# second arg is to let the evaluator know that it is training so it returns both accuracies
		train_accuracy, test_accuracy = evaluate_model(classifier=classifier, method="train")
		f.write(f'{learning_rate},{train_accuracy},{test_accuracy}\n')

		# save model
		torch.save(classifier.state_dict(), f'models/full_{argv[1]}_{round(learning_rate, 3)}.pth')

		# increase learning rate by 0.001 for next iteration
		learning_rate += 0.001

	f.close()
		

if __name__ == '__main__':
	main()