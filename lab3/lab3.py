#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import block
import torch
import time
import numpy as np
import seaborn

from torchvision import models
from torch import nn
from torch import optim
from torch.utils import data
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

from dataloader import RetinopathyLoader, getData

seaborn.set()


# In[2]:


# hyper parameter setting

batch_size = 32
epochs = 15
dataset_path = './data/'
device = torch.device("cuda")


# In[3]:


# dataset setup
train_dataset = RetinopathyLoader(dataset_path, 'train', device)
test_dataset = RetinopathyLoader(dataset_path, 'test', device)


train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = data.DataLoader(test_dataset, shuffle=True, batch_size=batch_size)


# In[6]:


# Resnet training
def train(model, optimizer, loss_func, epochs, model_name, scheduler=None):
    acc_train, acc_test = list(), list()
    acc_best = 0
    for epoch in range(epochs):
        start = time.time()
        epoch_loss = 0
        model.train()
        
        for i, batch_data in enumerate(train_loader):
            data, label = batch_data
            optimizer.zero_grad()
            
            predict = model(data)
            loss = loss_func(predict, label)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if scheduler is not None: 
            scheduler.step()
        
        model.eval()
        acc_train.append(test(model, train_loader)[0])
        acc_test.append(test(model, test_loader)[0])
        end = time.time()
        print(f'-------------------------[Epoch {epoch + 1}]-------------------------')
        print(f'loss: {epoch_loss}')
        print(f'Training Acc: {acc_train[-1]}')
        print(f'Testing Acc: {acc_test[-1]}\n')
        print(f'Time Used: {end - start}')
        for param_group in optimizer.param_groups:
            print(f"Current Learning Rate: {param_group['lr']}\n")
        
        if acc_test[-1] > acc_best:
            acc_best = acc_test[-1]
            torch.save(model.state_dict(), f'{model_name}_best.pth')
    
    return acc_train, acc_test

def test(model, test_loader):
    total, correct = 0, 0
    predict_res = list()
    with torch.no_grad():
        for test_data in test_loader:
            data, label = test_data
            predict = model.forward(data)
            _, predict = torch.max(predict, dim=1)
            
            total += label.size(0)
            correct += (predict == label).sum().item()
            
            predict = predict.cpu().numpy().tolist()
            predict_res += predict
            
    predict_res = np.array(predict_res)
    acc = correct / total * 100
    
    return acc, predict_res


# In[7]:


def resnet_training(model, model_name, layers=18):
    if layers == 18:
        if 'Pretrain' in model_name:
            model.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(in_features=512, out_features=128),
                nn.Linear(in_features=128, out_features=5)
            )
        else:
            model.fc = nn.Sequential( 
                nn.Linear(in_features=512, out_features=128),
                nn.Linear(in_features=128, out_features=5)
            )
    else:
        if 'Pretrain' in model_name:
            model.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(in_features=2048, out_features=512),
                nn.Linear(in_features=512, out_features=128),
                nn.Linear(in_features=128, out_features=5)
            )
        else:
            model.fc = nn.Sequential(
                nn.Linear(in_features=2048, out_features=512),
                nn.Linear(in_features=512, out_features=128),
                nn.Linear(in_features=128, out_features=5)
            )
    
    avail_gpus = torch.cuda.device_count()
    if avail_gpus > 1:
        print(f'{avail_gpus} GPUs available, enter Data parallel mode')
        model = nn.DataParallel(model)
    model = model.to(device)
    print(model)
    
    if 'Pretrain' in model_name:
        LR = 1e-3
    else:
        LR = 0.01
    print(LR)
    
    optimizer = optim.SGD(model.parameters(), momentum=0.9, weight_decay=1e-5, lr=LR, nesterov=True)
    loss_func = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 9], gamma=0.1)
    
    
    print('Starting Training')
    start = time.time()
    acc_train, acc_test = train(model=model, optimizer=optimizer, loss_func=loss_func, epochs=epochs, scheduler=scheduler, model_name=model_name)
    end = time.time()
    
    print(f'Training Process of ResNet{layers} took {end-start} secs in total')
    
    return acc_train, acc_test


# In[8]:


model_pretrained18 = models.resnet18(pretrained=True)
model_scratch18 = models.resnet18(pretrained=False)

model_pretrained50 = models.resnet50(pretrained=True)
model_scratch50 = models.resnet50(pretrained=False)

_, labels = getData('test')
classes = np.array([0, 1, 2, 3, 4])


# In[9]:


def plot_confusion(labels, predict, classes, model_name):
    confusion = confusion_matrix(labels, predict, classes, normalize='true')
    fig = plt.figure(figsize=(12, 8), dpi=300)
    
    ax = seaborn.heatmap(confusion, vmin=0, vmax=1, cmap=plt.cm.Blues, annot=True)
    ax.set_title(f'Normalized confusion matrix of {model_name}')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('Ground truth')
    plt.savefig(f'{model_name}_confusion.jpg')
    
    return
    


# In[10]:


def plot_comparison(model_name, accs):
    fig = plt.figure(figsize=(12, 8), dpi=300)
    plt.title(f'Result Comparison of {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
        
    for (acc_train, acc_test), pretrain in accs:
        plt.plot(acc_train, label=f'{model_name}_{pretrain}-Train', linewidth=0.8)
        plt.plot(acc_test, label=f'{model_name}_{pretrain}-Test', linewidth=0.8)
    plt.legend()
    plt.savefig(f'{model_name}_comparison.jpg')
    plt.show()
    
    return


# In[9]:


acc = list()

acc.append([resnet_training(model=model_pretrained18, layers=18, model_name='Pretrain18'), 'Pretrain18'])
final_acc, predict = test(model_pretrained18, test_loader)
print('Final Accuracy: ', final_acc)
print('Start processing Confusion Matrix')
plot_confusion(labels, predict, classes, 'Pretrained18')


# In[10]:


acc.append([resnet_training(model=model_scratch18, layers=18, model_name='Scratch18'), 'Scratch18'])
final_acc, predict = test(model_scratch18, test_loader)
print('Final Accuracy: ', final_acc)
print('Start processing Confusion Matrix')
plot_confusion(labels, predict, classes, 'Scratch18')

plot_comparison('ResNet18', acc)


# In[12]:


acc = list()
acc.clear()

acc.append([resnet_training(model=model_pretrained50, layers=50, model_name='Pretrain50'), 'Pretrain50'])
final_acc, predict = test(model_pretrained50, test_loader)
print('Final Accuracy: ', final_acc)
print('Start processing Confusion Matrix')
plot_confusion(labels, predict, classes, 'Pretrained50')


# In[13]:


# acc = list()
acc.append([resnet_training(model=model_scratch50, layers=50, model_name='Scratch50'), 'Scratch50'])
predict = test(model_scratch50, test_loader)[1]
plot_confusion(labels, predict, classes, 'Scratch50')

plot_comparison('ResNet50', acc)


# In[ ]:


model = models.resnet50(pretrained=False)
model.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=512),
        nn.Linear(in_features=512, out_features=128),
        nn.Linear(in_features=128, out_features=5)
    )
print(torch.load('Pretrain50_best.pth').module)
model.load_state_dict(torch.load('Pretrain50_best.pth').module.state_dict())
model = nn.DataParallel(model)
model = model.to(device)
acc, _ = test(model, test_loader)
        
print (acc)


# In[ ]:




