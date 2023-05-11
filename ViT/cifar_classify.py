import os
import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import csv
from model2 import ViT
from utils import progress_bar

NUM_EPOCHS = 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Running on", device)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    # transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    # transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='/home/niranjan.rajesh_ug23/AML/AML_Distracted_Drivers/data', train=True,
                                        download=False, transform=transform_train)

trainset = torch.utils.data.Subset(trainset, range(0,100))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='/home/niranjan.rajesh_ug23/AML/AML_Distracted_Drivers/data', train=False,
                                       download=False, transform=transform_test)

testset = torch.utils.data.Subset(testset, range(0,10))                                      
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)




classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')





vit =  ViT(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = int(512),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)


if 'cuda' in device:
    print(device)
    print("using data parallel")
    vit = torch.nn.DataParallel(vit) # make parallel
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit.parameters())
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS)

##### Training

def train(n_epochs):
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = inputs.to(device), targets.to(device)
            outputs = vit(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
        return train_loss/(batch_idx+1)
best_acc = 0  
def test(epoch):
    global best_acc
    vit.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = vit(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": vit.state_dict(),
              "optimizer": optimizer.state_dict()}
        if not os.path.isdir('/home/niranjan.rajesh_ug23/AML/AML_Distracted_Drivers/checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './home/niranjan.rajesh_ug23/AML/AML_Distracted_Drivers/checkpoint'+'vit'+'-ckpt.t7')
        best_acc = acc
    
    os.makedirs("/home/niranjan.rajesh_ug23/AML/AML_Distracted_Drivers/log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_vit.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss, acc




list_loss = []
list_acc = []


    
vit.to(device)
for epoch in range(NUM_EPOCHS):
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc = test(epoch)
    
    scheduler.step(epoch-1) # step cosine scheduling
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    

    # Write out csv..
    with open(f'log/log_vit.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    print(list_loss)


