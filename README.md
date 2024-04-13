# Folders (in order of generation)

## dataset

- Contains the original dataset, Wisconsin Diagnostic Breast Cancer (WDBC), from The University of Wisconsin
- Also contains a description of the dataset

## data

### 3 sub-folders named 'n_features' (where n is 10, 20, or 30)

- Contains pickle files that are serialized Pandas dataframes with the first n features of the dataset and their labels
- The data is split in a 70:30 ratio for training and testing, respectively
- Features and labels are stored in separate pickle files, as are training and testing data
  - This is done to ensure consistency, with the same rows being used for training and testing in all iterations

## models

- Contains all full models from training in the format 'full*{number of features}*{learning rate}.pth'
  - The best model is duplicated, and the learning rate is removed from the name for used in pruning
- The pruned model with maximum pruning, while maintaining >90% accuracy compared to the full model, is also included in the format 'pruned*{number of features}*{threshold}.pth'

## results

- Contains CSV files with the training accuracy, test accuracy, and learning rate for each full model in the format 'train\_{number of features}\_features.csv'
- Contains CSV files with the pruning threshold, accuracy against the test set, and model sparsity in the format 'pruned*accuracy*{number of features}\_features.csv'
  - Values are recorded as the pruning threshold is increased in incremments of 0.001 until accuracy drops below 90% relative to the full model

## plots

- Contains plots of sparsity vs. accuracy, threshold vs. accuracy, and threshold vs. sparsity for all pruned models

# Classes

## NeuralNet.py

- Definition of the model used in the program
  - Model has num_features input features (10, 20, or 30), which is passed as an initialization parameter
  - It is a sequential model with back-propogation of loss
  - Model has 2 hidden layers, both of which have the floor of num_features \* 2 / 3 neurons
    - Each hidden layer utilizes the ReLU activation function

## MagnitudePruner.py

- Extends the PyTorch base class of BasePruningMethod
- Performs unstructured pruning, which allows for pruning of individual weights
- The pruning magnitude threshold is set as an initialization parameter
- It creates a mask on the model for weights whose magnitudes fall below the pruning threshold, which are subsequently zeroed

# Functional Files

## create_dataset.py

- Takes the number of features to select as a command line argument
- Reads the full dataset into a Pandas dataframe and drops all features beyong the argument value
- Labels the remaining features by their name in the dataset
- Maps the labels, which are strings, to integers
  - 'M' for malignant becomes 1
  - 'B' for benign becomes 0
- Performs a random split of the dataset to remove any bias that may be present in the order
  - 70% of the dataset is used for training with the remaining 30% used for testing
  - There is no validation set due to the limited number of samples
- The labels are saves seperately from the features, but they are stored in the same order, ensuring they remain accurate
- Each dataset subset and it's labels are store in pickle files that can be read into a dataframe in other parts of the program
  - This ensures that the same values are always used for training and testing, removing that as a bias in training various models

## train_model.py

- The training features and labels are read into dataframes and converted to PyTorch tensors
  - The dataframes created a based on the number of features, passed as a command line argument
- These tensors are used to create a TensorDataset, which are loaded into a DataLoader, which can be batched during trianing
- A file is created to record the training performance using the Adam optimizer and training rates from 0.001-0.005
- During training, batch sizes of 32 are used and it is trained for 1000 epochs
  - After each batch, the loss is calculated for the batch, and it is backpropogated to the model
- After training, the model accuracy is calculated on both the training and testing datasets to check for overfitting, using the evaluate_model function, which will be described later
- Each model is saved, creating 15 models for 3 different feature counts and 5 different learning rates

## prune_model.py

- Run 3 times, once for each of the best full models for each feature count
- It takes the feature count as a command line arguments and selects the full model based on that argument
- Gets the test accuracy of the full model
- Defines the parameters to pruned, which are the weights from each hidden layer
- Creates a file to track accuracy and sparsity of the models as the pruning threshold is increased
  - The original accuracy is recorded as a threshold of 0
- With the pruning threshold beginning at 0.01, the threshold is iteratively raised by 0.001 until the accuracy of the pruned model drops below 90% of the original model accuracy on the test set.
  - The evaluate_model function is again used for accuracy testing
- Once the maximum threshold that keeps the model above 90% relatively accuracy is found, the pruned model is re-created and saved.

## evaluate_model.py

- This file is designed to be run standalone or called from another file
  - Initially, I was calling it manually, but I realized it was easier to build functionality to have other files call it to automate testing, so that is the functionality that will be covered
- The program runs differently based on the number of keyword agruments passed or if it's called from the command line
  - If only one keyword argument, 'classifer', is passed, it only returns the accuracy vs the test set
  - If two keyword arguments are passed, 'classifer' and anything else, it will return the accuracy against the training and testing datasets
  - The two keyword argument is used to evaluate the model during training, and the single keyword is used during pruning where only the accuracy against the test set is used
    - NOTE: Becuase the number of features is passed via command line for all other scripts calling this one, it can still read argv[1] to select the correct data
  - If called from the command line, the first argument is the number of features, second is either 'pruned' or 'full', and the third, which is only included if the second is 'pruned', is the threshold for the pruned model.
    - This is how the evaluator loads the correct model
- The data is loaded and converted to tensors, as it was in train_model.py
- The appropriate model is loaded, either from the arguments or based on command line input
- The accuracies are calculated by iterating through the feature and label tensors
  - The features tensor is used as input into thte classifier and the output is compared to the label tensor
  - If they match, the total_correct is incremented
  - At the end, the total_correct is divided by the total to get the percentage correct
- Depending on the arguments, either only the test accuracy is returned or a tuple of both the test and training accuracy is returned

## plot_results.py

- Used matplotlib to create plots of Pruning Threshold vs. Accuracy, Pruning Threshold vs. Sparsity, and Sparsity vs. Accuracy for each feature count, based on the value passed as a command line argument
- The data is from the files created in prune_model.py

# Overall Program Flow

1. create_dataset.py is called with command line arguments of 10, 20, and 30 to generate the dataframes of the dataset features and labels
2. train_model.py is called with command line arguments of 10, 20, and 30 to get full models for each feature count

- It calls evaluate_model.py to get training and test accuracies after training at each learning rate

3. prune_model.py is called with command line arguments of 10, 20, and 30 to generate pruned models with maximum pruning thresholds while maintaining accuracies > 90% relative to the full model

- It calls evaluate_model.py to get test accuracies after training for each pruning threshold

4. plot_results.py is called with command line arguments of 10, 20, and 30 to generate plots of the results from pruning
