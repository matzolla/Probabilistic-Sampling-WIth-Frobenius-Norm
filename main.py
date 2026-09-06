# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import get_dataloader
from model import video_Mvit, resnet3D
import argparse
import logging
from tqdm import tqdm

# Custom FileHandler that flushes after each log message
class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

# Set up logging with the custom handler
def setup_logging(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = FlushFileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def train(model, train_loader, criterion, optimizer,device):
    model.train()
    running_loss = 0.0
    correct=0.0
    total=0.0
    for _, param in model.named_parameters():
        param.requires_grad = False

    ##fine-tuning as we don't have enough resources for training from scratch
    for name, param in model.named_parameters():
       if name.startswith('r3d_18.layer4') or \
          name.startswith('r3d_18.fc'):
           param.requires_grad = True 
    model.to(device)
    for _, (data, target) in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        data,target=data.to(device),target.to(device)
        
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    avg_loss = running_loss / len(train_loader)
    avg_acc= correct/total
    
    return avg_loss,avg_acc

def main(args):
    # Set up logging
    
    logging.info('start')
    # Load data
    train_loader = get_dataloader(data_path=args.path,batch_size=args.batch_size,clip_length=8,num_clips=1,num_workers=args.num_workers)
    # Model, criterion, optimizer
    model =resnet3D(51).to(args.device) #get_resnet_3d_18_with_groupnorm().to(args.device) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        logging.info(f'Start training for epoch {epoch}')
        train_loss,avg_acc = train(model, train_loader, criterion, optimizer,args.device)
        logging.info(f'Epoch: {epoch}, Train Loss: {train_loss}')
        logging.info(f'Epoch: {epoch}, Accuracy  : {avg_acc}')
        # Save checkpoint
        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pth')
        logging.info(f'Checkpoint for epoch {epoch} saved.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UCF-101 DP-SGD Training')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=120, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--log_file', type=str, default='full_training_scheme.log', help='log file')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--path',type=str,default=r"C:\Users\23113181\Downloads\hmdb51_train_test_split_1\train",help='path to training data')
    args = parser.parse_args()
    setup_logging(args.log_file)
    main(args)