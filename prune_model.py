from NeuralNet import NeuralNet
import torch
from sys import argv
from torch.nn.utils import prune
from MagnitudePruner import MagnitudePruner
from evaluate_model import main as evalaute_model


# prune the model until the accuracy drops below 90% of the original accuracy
# the model is pruned in steps of 0.001
# the results are saved in a csv file called pruned_accuracy_{number of features}_features.csv for plotting
def main():
    # load the unpruned model
    classifier = NeuralNet(int(argv[1]))
    classifier.load_state_dict(torch.load(f'models/full_{int(argv[1])}.pth'))

    # evaluate the unpruned model
    orig_test_accuracy = evalaute_model(classifier=classifier)

    # set the parameters to prune for the MagnitudePruner
    parameters_to_prune = ((classifier.model.hl1, "weight"), (classifier.model.hl2, "weight"))

    # open a file called pruned_accuracy_{number of features}_features.csv to collect the results of the pruned models
    f = open(f'results/pruned_accuracy_{argv[1]}_features.csv', 'w')
    f.write('Threshold,Accuracy,Sparcity\n')
    f.write(f'0, {orig_test_accuracy},0\n')

    # initial threshold for pruning
    # pruned accuracy is set to 1 to start the loop
    pruned_accuracy = 1
    initial_threshold = 0.01

    # prune the model until the accuracy drops below 90% of the original accuracy with a step size of 0.01
    while pruned_accuracy > orig_test_accuracy * 0.9:
        # re-load the unpruned model
        classifier.load_state_dict(torch.load(f'models/full_{int(argv[1])}.pth'))

        # perform magnitude pruning
        prune.global_unstructured(parameters_to_prune, pruning_method=MagnitudePruner, threshold=initial_threshold)
        # remove weights with a 0 mask, having a 0 mask means the weight is less than the threshold
        prune.remove(classifier.model.hl1, "weight")
        prune.remove(classifier.model.hl2, "weight")
        
        # evaluate the pruned model
        pruned_accuracy = evalaute_model(classifier=classifier)
        # calculate the sparcity of the model
        sparcity = float(torch.sum(classifier.model.hl1.weight == 0) + torch.sum(classifier.model.hl2.weight == 0)) \
            / float(classifier.model.hl1.weight.nelement() + classifier.model.hl2.weight.nelement())
        # write the results to the file
        f.write(f'{round(initial_threshold, 3)},{pruned_accuracy},{round(sparcity, 3)}\n')

        # increase the threshold by 0.001 for next iteration
        initial_threshold += 0.001

    f.close()

    # save the most pruned model with accuracy above 90% of the original accuracy
    classifier.load_state_dict(torch.load(f'models/full_{int(argv[1])}.pth'))
    # threshold is decreased by 0.001 to get the model with accuracy above 90% of the original accuracy
    prune.global_unstructured(parameters_to_prune, pruning_method=MagnitudePruner, threshold=initial_threshold - 0.001)
    prune.remove(classifier.model.hl1, "weight")
    prune.remove(classifier.model.hl2, "weight")

    # save the most pruned model with accuracy above 90% of the original accuracy
    torch.save(classifier.state_dict(), f'models/pruned_{argv[1]}_{round(initial_threshold - 0.001, 3)}.pth')


if __name__ == '__main__':
    main()