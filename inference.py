# inference.py
import torch
import torch.nn as nn
from dataloader import get_dataloader
from model import video_Mvit,resnet3D
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

def test(model, test_loader, criterion,device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data,target=data.to(device),target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    logging.info(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {int(correct)}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    return test_loss, accuracy

## the loaded model comes with a "_module." on each key of the deeplearning layers. 
## this is because it was wrapped on the privacyengine module (I think). So we have to 
## remove it 
def load_model_with_mapped_keys(model, checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    # Get the state dictionary from the checkpoint
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Remove `_module.` prefix from state_dict keys
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('_module.', '')  # Remove the prefix
        new_state_dict[new_key] = v
    
    # Load the new state dictionary into the model
    model.load_state_dict(new_state_dict, strict=False)
    
    return model

# Example usage

def main(args):
    
    # Load data
    test_loader = get_dataloader(data_path=args.path,batch_size=args.batch_size,clip_length=8,num_clips=1,num_workers=args.num_workers)
    #print(len(test_loader))
    #print(len(test_loader.dataset))
    # Model, criterion
    model = resnet3D(51).to(args.device)#get_resnet_3d_18_with_groupnorm().to(args.device)
    #model = load_model_with_mapped_keys(model, args.model_path)    
    checkpoint=torch.load(args.model_path)
    #print(checkpoint.keys())
    model.load_state_dict(checkpoint)
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate
    test_loss, accuracy = test(model, test_loader, criterion,args.device)
    logging.info(f'Test Loss: {test_loss}, Accuracy: {accuracy}')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UCF-101 DP-SGD Inference')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for inference')
    parser.add_argument('--model_path', type=str, default=r"C:\Users\23113181\Desktop\softmax_segment_sampling\checkpoint_epoch_120.pth", help='path to the model checkpoint')
    parser.add_argument('--log_file', type=str, default='inference.log', help='log file')
    parser.add_argument('--device', type=str, default='cuda', help='device for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    parser.add_argument('--path',type=str,default=r"C:\Users\23113181\Downloads\hmdb51_train_test_split_1\test",help='path to testing data')
    args = parser.parse_args()
    setup_logging(args.log_file)
    main(args)
