import torch
from Model import NeuralNet
import pandas as pd
# import argv
from sys import argv

def main():
    test_features = pd.read_pickle('./data/test_features.pickle')
    test_labels = pd.read_pickle('./data/test_labels.pickle')

    # convert to tensors
    test_features = torch.tensor(test_features.to_numpy(), dtype=torch.float32)
    test_labels = torch.tensor(test_labels.to_numpy(), dtype=torch.long)

    # load model
    classifier = NeuralNet()
    classifier.load_state_dict(torch.load(f'models/{argv[1]}'))

    # calculate accuracy
    total_correct = 0
    total = len(test_labels)

    for i in range(len(test_features)):
        if test_labels[i] == torch.argmax(classifier(test_features[i])):
            total_correct += 1

    print(f"Accuracy: {total_correct / total}")

if __name__ == '__main__':
    main()