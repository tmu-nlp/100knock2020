import torch

class SingleLayerPerceptron(torch.nn.Module):
    def __init__(self, input_size, output_size):
        torch.manual_seed(1)
        super().__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=False)
        torch.nn.init.normal_(self.fc.weight, 0.0, 1.0)

    def forward(self, x):
        x = self.fc(x)
        return x

if __name__ == '__main__':
    X_train = torch.load('./tensors/X_train')
    model = SingleLayerPerceptron(X_train.shape[1], 4)

    y_hat_1 = torch.softmax(model.forward(X_train[0]), dim=-1)
    print(y_hat_1)

    Y_hat = torch.softmax(model.forward(X_train), dim=-1)
    print(Y_hat)


"""
tensor([0.0513, 0.6628, 0.0419, 0.2440], grad_fn=<SoftmaxBackward>)
tensor([[0.0513, 0.6628, 0.0419, 0.2440],
        [0.0141, 0.6940, 0.2675, 0.0244],
        [0.1783, 0.0302, 0.5850, 0.2065],
        ...,
        [0.1313, 0.5191, 0.0288, 0.3207],
        [0.0569, 0.2340, 0.5247, 0.1844],
        [0.4510, 0.0538, 0.2055, 0.2897]], grad_fn=<SoftmaxBackward>)
"""

