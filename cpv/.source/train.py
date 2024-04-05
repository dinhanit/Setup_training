import torch.optim as optim
import torch
from model import Custom_Model,Custom_Loss
from param import *
from dataloader import *
from tqdm import tqdm


def train_fn(model,data_loader,epochs,criterion,optimizer):
    for epoch in range(epochs):
            loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)
            
            for batch_idx, (inputs, targets) in loop:
                inputs = inputs.to(device).to(torch.float32)
                targets = targets.to(device)
                optimizer.zero_grad()
                out = model(inputs)
                loss = criterion(out, targets)
                loss.backward()
                optimizer.step()
                loop.set_postfix(loss=loss.item())
                
            checkpoint = {
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }
    return checkpoint
        
                
def utils_train(model,data_loader,epochs,criterion,optimizer,out_dir="",train_checkpoint = False,**kwargs):
    
    if train_checkpoint:
        checkpoint = torch.load(kwargs['checkpoint_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss'] 

        
    ckpt = train_fn(model,data_loader,epochs,criterion,optimizer)         
    torch.save(ckpt, f'{out_dir}/checkpoint.pth')
    print("Checkpoint saved for epoch:", epochs)
        
if __name__ =="__main__":
    model = Custom_Model()
    criterion = Custom_Loss()
    device = torch.device('cpu')

    model.to(device)
    criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(LR))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    
    utils_train(model=model,data_loader=TRAINLOADER,epochs=20,criterion=criterion,optimizer=optimizer,out_dir="../.model")
            
            