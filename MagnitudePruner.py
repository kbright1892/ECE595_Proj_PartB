from torch.nn.utils import prune
import torch


class MagnitudePruner(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_mask(self, tensor, default_mask):
        return torch.abs(tensor) > self.threshold
    

    # https://stackoverflow.com/questions/61629395/how-to-prune-weights-less-than-a-threshold-in-pytorch