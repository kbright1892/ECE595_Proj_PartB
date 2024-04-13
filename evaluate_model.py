import torch
from NeuralNet import NeuralNet
import pandas as pd
from sys import argv
from torch.nn.utils import prune

def main(**kwargs):
    # load data
    # train_features = pd.read_pickle(f'./data/{argv[1]}_features/train_features.pickle')
    # train_labels = pd.read_pickle(f'./data/{argv[1]}_features/train_labels.pickle')
    test_features = pd.read_pickle(f'./data/{argv[1]}_features/test_features.pickle')
    test_labels = pd.read_pickle(f'./data/{argv[1]}_features/test_labels.pickle')

    # convert to tensors
    # train_features = torch.tensor(train_features.to_numpy(), dtype=torch.float32)
    # train_labels = torch.tensor(train_labels.to_numpy(), dtype=torch.long)
    test_features = torch.tensor(test_features.to_numpy(), dtype=torch.float32)
    test_labels = torch.tensor(test_labels.to_numpy(), dtype=torch.long)

    classifier: NeuralNet = None

    # can be called independently and given a model via argv or called by another function and given a model via kwargs
    if len(kwargs) == 0:
        # load model
        classifier = NeuralNet(int(argv[1]))

        if len(argv) == 4: # pruned model
            classifier.load_state_dict(torch.load(f'models/{argv[2]}_{argv[1]}_{argv[3]}.pth'), strict=False)
        else: # unpruned model
            classifier.load_state_dict(torch.load(f'models/{argv[2]}_{argv[1]}.pth'), strict=False)
    else:
        classifier = kwargs['classifier']


    # calculate accuracy
    # total_correct = 0
    
    # total = len(train_labels)

    # for i in range(len(train_features)):
    #     if train_labels[i] == torch.argmax(classifier(train_features[i])):
    #         total_correct += 1

    # print(f"Train Accuracy: {total_correct / total}")

    total_correct = 0

    total = len(test_labels)

    for i in range(len(test_features)):
        if test_labels[i] == torch.argmax(classifier(test_features[i])):
            total_correct += 1

    test_accuracy = total_correct / total

    if len(kwargs) > 0:
        return test_accuracy
    else:
        print(f"Test Accuracy: {test_accuracy}")

if __name__ == '__main__':
    main()