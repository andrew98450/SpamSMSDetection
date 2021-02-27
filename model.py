import torch

class SpamNet(torch.nn.Module):
    
    def __init__(self, vocab_size):
        super(SpamNet, self).__init__()
        self.embedding_layer = torch.nn.Embedding(vocab_size, 300)
        self.lstm_layer = torch.nn.LSTM(300, 500, batch_first=True)
        self.fc_layer = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(50 * 500, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 2),
            torch.nn.Softmax(1))
    
    def forward(self, inputs):
        outputs = self.embedding_layer(inputs)
        outputs, _ = self.lstm_layer(outputs)
        outputs = self.fc_layer(outputs)
        return outputs