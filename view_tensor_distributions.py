import matplotlib.pyplot as plt
import torch
from NeuralNet import NeuralNet
from sys import argv
import numpy as np

# takes the number of features as a command line argument
# saves a histogram of the weights of both hidden layers
def main():
    # load the neural network
    classifier = NeuralNet(int(argv[1]))
    classifier.load_state_dict(torch.load(f'models/full_{argv[1]}.pth'))
    
    # get the weights from the first layer
    weights_hl1 = classifier.model.hl1.weight.detach().numpy()
    weights_hl2 = classifier.model.hl2.weight.detach().numpy()

    weights = np.concatenate((weights_hl1.flatten(), weights_hl2.flatten()))
    
    # plot the distribution of the weights
    plt.rcParams["figure.figsize"] = [5, 3]
    plt.rcParams["figure.autolayout"] = True
    plt.hist(weights, bins=50)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title(f'Weight Distribution of Full Model with {argv[1]} Features')
    plt.savefig(f'plots/weights_{argv[1]}.png')
    plt.clf()

if __name__ == '__main__':
    main()
