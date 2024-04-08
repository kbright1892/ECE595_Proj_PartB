from torch import nn, cuda

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