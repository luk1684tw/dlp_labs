#!/usr/bin/env python
# coding: utf-8

# In[64]:


# import modules needed
import torch
from matplotlib import pyplot as plt

from dataloader import read_bci_data


# In[65]:


# hyper parameter setup
batch_size = 64
lr = 0.001
epochs = 1000
print_interval = 100
activation_list = {
    'Relu': torch.nn.ReLU(),
    'LeakyRelu': torch.nn.LeakyReLU(),
    'ELU': torch.nn.ELU()
}

# torch setup and dataset handler
device = torch.device("cuda:1")
torch.manual_seed(87)


# In[66]:


# Custom Dataset Implementation
class BCIDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = torch.from_numpy(data).type(torch.FloatTensor).to(device)
        self.label = torch.from_numpy(label).type(torch.LongTensor).to(device)

        return
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        
        return data, label

train_data, train_label, test_data, test_label = read_bci_data()

train_dataset = BCIDataset(train_data, train_label)
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = BCIDataset(test_data, test_label)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True, batch_size=batch_size)


# In[67]:


# EEGNet implementation
class EEGNet(torch.nn.Module):
    def __init__(self, activation_mode):
        super(EEGNet, self).__init__()
        self.Conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            torch.nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.DepthWiseConv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation_list[activation_mode],
            torch.nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            torch.nn.Dropout(p=0.25)
        )
        self.SeparableConv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), groups=16, bias=False),
            torch.nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation_list[activation_mode],
            torch.nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            torch.nn.Dropout(p=0.25)
        )
        self.Classification = torch.nn.Linear(in_features=736, out_features=2, bias=True)
        
    def forward(self, x):
        bs = x.shape[0]
        x = self.Conv(x)
        x = self.DepthWiseConv(x)
        x = self.SeparableConv(x)
        x = x.view(bs, -1)
        
        return self.Classification(x)


# In[68]:


# DeepConvNet implementation
class DeepConvNet(torch.nn.Module):
    def __init__(self, activation_mode):
        super(DeepConvNet, self).__init__()
        self.Layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 5)),
            torch.nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(2, 1)),
            torch.nn.BatchNorm2d(num_features=25, eps=1e-05, momentum=0.1),
            activation_list[activation_mode],
            torch.nn.MaxPool2d(kernel_size=(1, 2)),
            torch.nn.Dropout2d(p=0.5)
        )
        
        self.Layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1, 5)),
            torch.nn.BatchNorm2d(num_features=50, eps=1e-05, momentum=0.1),
            activation_list[activation_mode],
            torch.nn.MaxPool2d(kernel_size=(1, 2)),
            torch.nn.Dropout2d(p=0.5)
        )
        
        self.Layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1, 5)),
            torch.nn.BatchNorm2d(num_features=100, eps=1e-05, momentum=0.1),
            activation_list[activation_mode],
            torch.nn.MaxPool2d(kernel_size=(1, 2)),
            torch.nn.Dropout2d(p=0.5)
        )
        
        self.Layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(1, 5)),
            torch.nn.BatchNorm2d(num_features=200, eps=1e-05, momentum=0.1),
            activation_list[activation_mode],
            torch.nn.MaxPool2d(kernel_size=(1, 2)),
            torch.nn.Dropout2d(p=0.5)
        )
        
        self.classifier = torch.nn.Linear(in_features=8600, out_features=2, bias=True)
        
        return
    
    def forward(self, x):
        bs = x.shape[0]
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = self.Layer4(x)
        
        x = x.view(bs, -1)
        
        return self.classifier(x)
        
        
        


# In[69]:


# Training/Testing function Implementaion
def train(model, optimizer, loss_func, scheduler=None):
    acc_train, acc_test = list(), list()
    for epoch in range(epochs):
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
        acc_train.append(test(model, train_loader))
        acc_test.append(test(model, test_loader))
        
        if epoch % print_interval == 0:
            print(f'-------------------------[Epoch {epoch}]-------------------------')
            print(f'loss: {epoch_loss}')
            print(f'Training Acc: {acc_train[-1]}')
            print(f'Testing Acc: {acc_test[-1]}\n')
#             for param_group in optimizer.param_groups:
#                 print(f"Current Learning Rate: {param_group['lr']}")
            
    return acc_train, acc_test


def test(model, test_loader):
    total, correct = 0, 0
    with torch.no_grad():
        for test_data in test_loader:
            data, label = test_data
            predict = model.forward(data)
            _, predict = torch.max(predict, dim=1)
            
            total += label.size(0)
            correct += (predict == label).sum().item()
    
    acc = correct / total * 100
    
    return acc
    


# In[70]:


# Model training entry
def train_entry(model_name, activation):
    if model_name is 'EEG':
        model = EEGNet(activation).to(device)
    else:
        model = DeepConvNet(activation).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss()

    # Decaying lr with a factor 0.95 every 25 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.95)

    acc_train, acc_test = train(model, optimizer, loss_func, scheduler)
    print('Final Acc:', acc_test[-1])
        
    return acc_train, acc_test


# In[71]:


# Plot Function Implementation
def plot_comparison(model_name):
    fig = plt.figure(figsize=(12, 8), dpi=300)
    plt.title(f'Activation Function Comparison of {model_name} Net')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
        
    for activation_func in activation_list:
        if model_name is 'EEG':
            acc_train, acc_test = train_entry('EEG', activation_func)
        else:
            acc_train, acc_test = train_entry('DCN', activation_func)
    
        plt.plot(acc_train, label=f'{activation_func}-Train', linewidth=0.8)
        plt.plot(acc_test, label=f'{activation_func}-Test', linewidth=0.8)
    plt.legend()
    plt.savefig(f'{model_name}_comparison.jpg')
    plt.show()


# In[72]:


plot_comparison('EEG')


# In[73]:


plot_comparison('DCN')

