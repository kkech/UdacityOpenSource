import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms


class Data_client(Dataset):
    
    def __init__(self, num_classes=3, samples=1000, device="cuda"):
        
        super().__init__()
        
        self.samples = samples
        self.device = device
        self.num_classes = num_classes
        
        transform = transforms.Compose([transforms.ToTensor()])
        self.mnist_trainset = dsets.MNIST(root='./data', train=True, download=True, transform=transform)
        
        self.data = []
        self.l = []
            
        for i in range(0, num_classes):
            
            data = []
            sample_count = 0
            
            for datum in self.mnist_trainset:
                if datum[1] == i:
                    data.append(datum)
                    sample_count+=1
                    
                if sample_count == samples : 
                    break
                    
            self.data.append(data) 
            self.l.append( torch.tensor(np.zeros(1,dtype=np.float32)+i , dtype=torch.float32, device=device) )
            
        return
    
    def __getitem__(self, inx):
        
        rand = np.random.randint(0, high=self.samples)
        
        return [ (self.data[i][inx][0].to(self.device), self.data[i][rand][0].to(self.device), self.l[i]) for i in range(0, self.num_classes) ]
    
    def __len__(self):
        
        return len(self.data[0])
    
    
    def client_sampler(self, client, samples, dataset):
    
        f = lambda x: True if client == x[1] else False

        return list( filter(f, dataset) )[:samples]
    
    
    
class Data_server(Dataset):
    
    def __init__(self, biomarkers, models):
        
        super(Data_server,self).__init__()
    
        self.model = models
        num_clients = len(models)
        
        self.data = biomarkers
        dataloader = DataLoader(biomarkers, batch_size=64, shuffle=True) 
    
        self.encrypted_data = []
        self.labels = []
        
        with torch.no_grad():
                
            for data in dataloader :
                
                out = [ self.model[i].forward_once( data[i][0] ) for i in range(0, num_clients) ]
                M = [ data[i][2] for i in range(0, num_clients) ]

                labels = torch.cat( M, dim=0)
                M_ = torch.cat( out, dim=0)
                
                self.encrypted_data.append(M_)
                self.labels.append(labels)

            self.labels = torch.cat(self.labels).squeeze()
            self.encrypted_data = torch.cat(self.encrypted_data,dim=0).squeeze()

        return
    
    
    def __getitem__(self, inx): 
        return [ self.encrypted_data[inx], self.labels[inx] ]
    
    
    def __len__(self):    
        return len(self.encrypted_data)
   

    
class Data_per_client(Dataset):
    
    def __init__(self, biomarkers, model, num_clients):
        
        super().__init__()
        
        self.data = biomarkers
        dataloader = DataLoader(biomarkers, batch_size=64, shuffle=True) 
    
        self.encrypted_data = []
        self.labels = []
        
        with torch.no_grad():
                
            for data in dataloader :
                
                out = [ model.forward_once( data[i][0] ) for i in range(0, num_clients) ]
                M = [ data[i][2] for i in range(0, num_clients) ]

                labels = torch.cat( M, dim=0)
                M_ = torch.cat( out, dim=0)
                
                self.encrypted_data.append(M_)
                self.labels.append(labels)

            self.labels = torch.cat(self.labels).squeeze()
            self.encrypted_data = torch.cat(self.encrypted_data,dim=0).squeeze()
  
        return


    def __getitem__(self, inx):
        return [self.encrypted_data[inx], self.labels[inx]]

    
    def __len__(self):
        return len(self.encrypted_data)