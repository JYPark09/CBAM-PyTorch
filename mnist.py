import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from model.model import Network

import time

USE_CUDA = torch.cuda.is_available()

EPOCHS = 40
BATCH_SIZE = 128
LEARNING_RATE = 1e-1
WEIGHT_DECAY = 1e-4

def main():
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    net = Network(1, 128, 10, 10)

    if USE_CUDA:
        net = net.cuda()

    opt = optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=.9, nesterov=True)

    for epoch in range(1, EPOCHS + 1):
        print('[Epoch %d]' % epoch)
        
        train_loss = 0
        train_correct, train_total = 0, 0

        start_point = time.time()

        for inputs, labels in train_loader:
            inputs, labels = Variable(inputs), Variable(labels)
            if USE_CUDA:
                inputs, labels = inputs.cuda(), labels.cuda()

            opt.zero_grad()

            preds = F.log_softmax(net(inputs), dim=1)
            
            loss = F.cross_entropy(preds, labels)
            loss.backward()

            opt.step()

            train_loss += loss.item()

            train_correct += (preds.argmax(dim=1) == labels).sum().item()
            train_total += len(preds)

        print('train-acc : %.4f%% train-loss : %.5f' % (100 * train_correct / train_total, train_loss / len(train_loader)))
        print('elapsed time: %ds' % (time.time() - start_point))

        test_loss = 0
        test_correct, test_total = 0, 0

        for inputs, labels in test_loader:
            with torch.no_grad():
                inputs, labels = Variable(inputs), Variable(labels)

                if USE_CUDA:
                    inputs, labels = inputs.cuda(), labels.cuda()

                preds = F.softmax(net(inputs), dim=1)

                test_loss += F.cross_entropy(preds, labels).item()

                test_correct += (preds.argmax(dim=1) == labels).sum().item()
                test_total += len(preds)

        print('test-acc : %.4f%% test-loss : %.5f' % (100 * test_correct / test_total, test_loss / len(test_loader)))
        
        torch.save(net.state_dict(), './checkpoint/checkpoint-%04d.bin' % epoch)

if __name__ == '__main__':
    main()