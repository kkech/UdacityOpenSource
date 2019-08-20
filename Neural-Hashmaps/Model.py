import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    
    def __init__(self):
        
        super(Encoder,self).__init__()
            
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1)  
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.linear1 = nn.Linear(400,256)
        self.linear2 = nn.Linear(256,64)
        self.linear3 = nn.Linear(64,2)
        
        return
        
        
    def forward_once(self,inp):

        out = F.relu(self.conv1(inp))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        out = self.pool(out)           

        out = out.view(-1,400)
        
        out = F.relu(self.linear1(out)) 
        out = F.relu(self.linear2(out))
        out = self.linear3(out)

        return out
    
    def forward(self, anc, pos):
        
        anc = self.forward_once(anc)  
        pos = self.forward_once(pos)  
        
        return (anc, pos)


class Discriminator(nn.Module):
    
    def __init__(self, num_classes=3, in_features=2):
        
        super(Discriminator,self).__init__()
                  
        self.linear1 = nn.Linear(in_features,64)
        self.linear2 = nn.Linear(64,256)
        self.linear3 = nn.Linear(256,64)
        self.linear4 = nn.Linear(64, num_classes)
        
        return
        
        
    def forward(self,inp):

        out = F.relu(self.linear1(inp))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = self.linear4(out)

        return out
    
    
class ContrastiveLoss(nn.Module):
 
    def __init__(self, margin=10.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x, y, label):
        
        euclidean_distance = nn.functional.pairwise_distance(x,y)
        
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
    
    
    
def get_client_models(number, device="cuda"):
    
    return [ Encoder().to(device) for i in range(0, number) ]

def test(model, dataloader):
    
    length = 0
    accuracy = 0

    
    for data in dataloader :
        
        img, lbl = data
        length+=len(lbl)    
        
        with torch.no_grad():

            preds = model(img)
            top_p, top_class = torch.topk(preds, 1, dim=1)
            accuracy += ( top_class == lbl.long().view(*top_class.shape) ).sum()
            
            
    accuracy = 100 * accuracy/length
   
    return accuracy