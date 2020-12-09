import numpy as np
import torch
import os
import glob
from torchvision import transforms
from torch.utils.data import Dataset
import time
import torch.nn as nn 
import torchvision 
import torch.nn.functional as F 
import matplotlib.pyplot as plt
import pandas as pd


class FontDataset(Dataset):
    def __init__(self, npy_dir, max_dataset_size=float("inf")):
        self.dir_path = npy_dir
        self.to_tensor = transforms.ToTensor()
        entry = []
        files = glob.glob1(npy_dir, '*npy')
        for f in files:
            f = os.path.join(npy_dir, f)
            entry.append(f)
        self.npy_entry = entry[:min(max_dataset_size, len(entry))]



    def __getitem__(self, index):
        npy_entry = self.npy_entry
        single_npy_path = npy_entry[index]
        single_npy = np.load(single_npy_path, allow_pickle=True)[0]
        single_npy_tensor = self.to_tensor(single_npy)
        single_npy_label = np.load(single_npy_path, allow_pickle=True)[1]
        return (single_npy_tensor, single_npy_label)



    def __len__(self):
        return len(self.npy_entry)

train_dir = './npy_train'
val_dir = './npy_val'
train_dataset = FontDataset(train_dir)
val_dataset = FontDataset(val_dir)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset , batch_size=100,shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset , batch_size=5000,shuffle=True)

num_epochs=15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Cha(nn.Module):
    def __init__(self,inchannel,outchannel,s):
        super(Cha,self).__init__()
        self.maxpool=nn.MaxPool2d(s)
        self.avgpool=nn.AvgPool2d(s)
        self.linear1=nn.Linear(outchannel,int(outchannel))
        self.linear2=nn.Linear(int(outchannel),outchannel)
        self.sigmoid=nn.Sigmoid()
        self.outchannel=outchannel
        
    def forward(self,x):
        m=self.maxpool(x)
        a=self.avgpool(x)
        m=m.reshape(m.size(0),-1)
        a=a.reshape(a.size(0),-1)
        m_=self.linear2(F.relu(self.linear1(m)))
        a_=self.linear2(F.relu(self.linear1(a)))
        t=m_+a_
        t=t.reshape(t.size(0),self.outchannel,1,1)
        return self.sigmoid(t)


class Spa(nn.Module):
    def __init__(self,inchannel,outchannel,s):
        super(Spa,self).__init__()
        self.maxpool=nn.MaxPool3d((outchannel,1,1))
        self.avgpool=nn.AvgPool3d((outchannel,1,1))
        self.conv=nn.Conv2d(2,1,7,stride=1,padding=3)
        
    def forward(self,x):
        m=self.maxpool(x)
        a=self.avgpool(x)
        r=self.conv(torch.cat([m,a],dim=1))
        
        return r
class BasicConv(nn.Module):
    def __init__(self,inchannel,outchannel,s):
        super(BasicConv,self).__init__()
        self.conv=nn.Conv2d(inchannel,outchannel,s,stride=1,padding=int((s-1)/2))
        self.ba=nn.BatchNorm2d(outchannel)
    def forward(self,x):
        x=self.conv(x)
        x=self.ba(x)
        return F.relu(x)
    
        
class AttentionConv(nn.Module):
    def __init__(self):
        super(AttentionConv,self).__init__()
        self.conv1=BasicConv(3,3,5)
        self.conv2=BasicConv(3,3,5)
        self.cha1=Cha(3,3,32)
        self.spa1=Spa(3,3,32)
        self.cha2=Cha(3,3,16)
        self.spa2=Spa(3,3,16)
        self.maxpool=nn.MaxPool2d(2)
        self.linear=nn.Linear(8*8*3,50)
        self.softmax=nn.Softmax()
    def forward(self,x):
        res=x
        x=self.conv1(x)
        x=self.cha1(x)*x
        x=self.spa1(x)*x
        
        x+=res
        
        
        x=self.maxpool(x)
        
        res2=x
        
        x=self.conv2(x)
        x=self.cha2(x)*x
        x=self.spa2(x)*x
        
        x+=res2
        
        x=self.maxpool(x)
        
        x=x.reshape(x.size(0),-1)
        x=self.linear(x)
        return x
    
    
model=AttentionConv().to(device)
    
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

images_, labels_=next(iter(val_loader))
import time

## 15 epoch    370초

if __name__ == '__main__': 
    start_time=time.time()
    total_step = len(train_loader) 
    for epoch in range(num_epochs): 
        for i, (images, labels) in enumerate(train_loader): 
            # Assign Tensors to Configured Device 
            images = images.to(device) 
            labels = labels.to(device) 
            

            # Forward Propagation 
            outputs = model(images) 
            
            loss = criterion(outputs, labels) 
            
            
            
            
            _, predicted = torch.max(outputs.data, 1) 
            
            total=labels.size(0)
            correct=(predicted==labels).sum().item()
            # Get Loss, Compute Gradient, Update Parameters 
            tr_acc=correct*100/total
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc: {:.2f}%'.format(epoch+1, num_epochs, i+1, total_step, loss.item(),tr_acc))
            
            if tr_acc>90:
                val_total=0
                val_correct=0
                images_=images_.to(device)
                labels_=labels_.to(device)
                outputs_=model(images_)
                _, pre=torch.max(outputs_.data,1)
                val_total=labels_.size(0)
                val_correct=(pre==labels_).sum().item()
                acc=val_correct*100/val_total
            
            
                if acc>max_acc:
                    max_acc=acc
                    print('max_acc:',acc,'%')
                    torch.save(model.state_dict(),'best_model(final).pth')
            
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 

    du=time.time()-start_time
    print(du)        

 
    
