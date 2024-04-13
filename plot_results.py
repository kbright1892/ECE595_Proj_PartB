from sys import argv
import matplotlib.pyplot as plt
import pandas as pd


# plot results uses the results from the pruned_accuracy_{number of features}_features.csv file to create graphs of the results
def main():
    # the number of features is used to read the correct csv file
    num_features = int(argv[1])
    
    # read the resulst csv file into a dataframe
    df = pd.read_csv(f'results/pruned_accuracy_{argv[1]}_features.csv', index_col=None, usecols=["Threshold", "Accuracy", "Sparsity"], sep=',')

    # plot the results

    # Pruining Threshold vs Accuracy
    plt.rcParams["figure.figsize"] = [8, 5]
    plt.rcParams["figure.autolayout"] = True
    plt.plot(df['Threshold'], df['Accuracy'])
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title(f'Pruning Threshold vs Accuracy with {num_features} features')
    plt.savefig(f'plots/threshold_accuracy_{argv[1]}_features.png')
    plt.clf()

    # Pruning Threshold vs Sparsity
    plt.plot(df['Threshold'], df['Sparsity'])
    plt.xlabel('Threshold')
    plt.ylabel('Sparsity')
    plt.title(f'Pruning Threshold vs Sparsity with {num_features} features')
    plt.savefig(f'plots/threshold_sparsity_{argv[1]}_features.png')
    plt.clf()

    # Sparsity vs Accuracy
    plt.plot(df['Sparsity'], df['Accuracy'])
    plt.xlabel('Sparsity')
    plt.ylabel('Accuracy')
    plt.title(f'Sparsity vs Accuracy with {num_features} features')
    plt.savefig(f'plots/sparsity_accuracy_{argv[1]}_features.png')
    plt.clf()


if __name__ == '__main__':
    main()