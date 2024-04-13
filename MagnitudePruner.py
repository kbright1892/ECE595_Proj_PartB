from torch.nn.utils import prune
import torch


# pruner that removes weights with magnitude less than a threshold
class MagnitudePruner(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    # default_mask is not used, but it is required by the base class
    # creates a mask that is 1 for weights with magnitude greater than the threshold, and 0 otherwise
    # when remove is called, the weights with a 0 in the mask are removed
    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold