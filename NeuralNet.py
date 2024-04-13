from torch import nn, cuda

# check if cuda is available, if not use cpu
# cuda is not avalable on my machine, but may be on another
processing_device = 'cuda' if cuda.is_available() else 'cpu'


# a basic linear network with 2 hidden layers that uses ReLU activation
# the input layer has num_features neurons
# the hidden layers have num_features * 2 // 3 neurons. 
# I couldn't find a great rule of thumb for the number of neurons in the hidden layers, so I followed some advice I found online
# the output layer has 2 neurons for binary classification
class NeuralNet(nn.Module):
	def __init__(self, num_features: int):
		super().__init__()

		self.model = nn.Sequential()

		self.model.add_module('hl1', nn.Linear(in_features=num_features, out_features=num_features * 2 // 3, device=processing_device))
		self.model.add_module('hl1_relu', nn.ReLU())
		self.model.add_module('hl2', nn.Linear(in_features=num_features * 2 // 3, out_features=num_features * 2 // 3, device=processing_device))
		self.model.add_module('hl2_relu', nn.ReLU())
		self.model.add_module('out', nn.Linear(in_features=num_features * 2 // 3, out_features=2, device=processing_device))


	def forward(self, x):
		return self.model(x)