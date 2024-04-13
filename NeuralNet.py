from torch import nn, cuda

# check if cuda is available, if not use cpu
# cuda is not avalable on my machine, but may be on another
processing_device = 'cuda' if cuda.is_available() else 'cpu'

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