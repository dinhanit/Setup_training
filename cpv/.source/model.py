import torch
import torch.nn as nn
from config import *

input_features = (224,224,3)
class Custom_Model(nn.Module):
    def __init__(self, num_classes=2):
        super(Custom_Model, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels=224,out_channels=112,kernel_size=3)
        self.conv1x1 = nn.Conv2d(in_channels=112,out_channels=64,kernel_size=1)
        self.linear = nn.Linear(in_features= 64*222, out_features= num_classes)
        
    def forward(self,x):
        b = x.shape[0]
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        x = x.view(b,-1)
        x = self.linear(x)
        return x
    
def test_model():
    input_test = torch.rand((8,224,224,3))
    model = Custom_Model()
    output_test = model(input_test)
    print(output_test.shape)
    
    
    
import torch
import torch.nn as nn

class Custom_Loss(nn.Module):
    def __init__(self):
        super(Custom_Loss, self).__init__()

    def forward(self, input, target):
        log_probs = torch.log_softmax(input, dim=1)
        loss = -log_probs.gather(dim=1, index=target.view(-1, 1))
        
        loss = loss.mean()
        return loss
    
def test_loss():
    input_tensor = torch.randn(3, 5)  # Example input tensor with batch size 3 and 5 classes
    target_tensor = torch.tensor([1, 0, 3])  # Example target tensor with class indices

    # Instantiate the custom loss
    custom_loss = Custom_Loss()

    # Compute the loss
    loss = custom_loss(input_tensor, target_tensor)
    print(loss)