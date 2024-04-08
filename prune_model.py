from Model import NeuralNet
import torch

def main():
    full_model = NeuralNet()
    full_model.load_state_dict(torch.load('models/full_model.pth'))

    for param in full_model.parameters():
        print(param)

if __name__ == '__main__':
    main()