
import torch.nn as nn
import torchvision

def video_Mvit():
    model= torchvision.models.video.mvit_v1_b(pretrained=True)
    num_ftrs=model.head[1].in_features
    model.head[1]=nn.Linear(num_ftrs,51)
    return model

class resnet3D(nn.Module):
    def __init__(self, num_classes):
        super(resnet3D, self).__init__()

        # Define R3D_18 as a submodule
        self.r3d_18 = torchvision.models.video.r3d_18(weights='R3D_18_Weights.KINETICS400_V1')

        # Modify the final fully connected layer for the desired number of classes
        self.r3d_18.fc = nn.Linear(self.r3d_18.fc.in_features, num_classes)

    def forward(self, x):
        # Forward pass through R3D_18 model
        x = self.r3d_18(x)

        return x