import torch
import numpy
from utils import load_dataset
from model import SpamNet

train_set, test_set, text_field, train_len, test_len, vocab_len = load_dataset(batch_size=1)

model = torch.load("spam_model.pth")
labels = ["No Spam", "Spam"]
for inputs in test_set:
    text = ""
    for i in range(50):
        text += text_field.vocab.itos[inputs.data.squeeze()[i]] + " "
    
    output = model(inputs.data)
    index = output.argmax(1).item()
    print("%s | %s" % (labels[index], text))