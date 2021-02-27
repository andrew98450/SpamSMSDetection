import torch
import argparse
from model import SpamNet
from utils import load_dataset

parse = argparse.ArgumentParser()
parse.add_argument("--dataset", default="Dataset/spam.csv", type=str)
parse.add_argument("--epoch", default=5, type=int)
parse.add_argument("--gpu", default=False, type=bool)
parse.add_argument("--batchsize", default=64, type=int)
parse.add_argument("--lr", default=0.001, type=float)
args = parse.parse_args()

dataset = args.dataset
batch_size = args.batchsize
epoch = args.epoch
use_gpu = args.gpu
lr = args.lr

train_set, test_set, text_field, train_len, test_len, vocab_len = load_dataset(
    dataset, 
    batch_size = batch_size)

spamnet = SpamNet(vocab_len)
spamnet = spamnet.cuda() if use_gpu else spamnet.cpu()
optim = torch.optim.Adam(params=spamnet.parameters(), lr=lr)
loss_f = torch.nn.NLLLoss()

for i in range(epoch):
    acc = 0.0
    loss = 0.0
    step = 0
    for datas in train_set:
        inputs = datas.data.long() 
        labels = datas.label.long() 

        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = torch.log(spamnet(inputs)) 
        error = loss_f(outputs, labels)

        optim.zero_grad()
        error.backward()
        optim.step()
        
        acc += torch.argmax(outputs, 1).eq(labels).sum().item()
        loss += error.item()
        step += 1
    
    acc /= train_len
    loss /= step
    print("[+] Epoch: %d Acc: %.4f Loss: %.4f" % (i + 1, acc, loss))

spamnet = spamnet.cpu().eval()
    
torch.save(spamnet, "spam_model.pth")