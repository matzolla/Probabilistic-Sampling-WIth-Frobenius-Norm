# dataloader.py
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import cv2
import torch
from tqdm import tqdm
from softmax_segment_sampler import soft_max_probabilities
import random


class UCF101ClipDataset(Dataset):
    def __init__(self, root_dir, clip_length=8,num_clips=1,transform=None):
        self.root_dir = root_dir
        self.clip_length = clip_length
        self.transform = transform
        self.num_clips=num_clips
        self.samples = self._make_dataset()
        self.classes = os.listdir(self.root_dir)
        self.classes.sort()

    # Create a mapping from class names to indices
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
    def _make_dataset(self):
        samples = []
        for class_dir in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                for video in os.listdir(class_path):
                    video_path = os.path.join(class_path, video)
                    samples.append((video_path, class_dir))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, class_label = self.samples[idx]
        clips = self._load_video_clips(video_path)
        probabilities= soft_max_probabilities(video_path)
        samples = random.choices(clips, weights=probabilities, k=4)

        labels= [int(self.class_to_idx[class_label])]*len(samples)
        return samples, labels

    def _load_video_clips(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(self.transform(frame))
        cap.release()
        
        clips = self._generate_clips(frames)
        return clips

    def _generate_clips(self, frames):
        clips = []
        total_frames = len(frames)

        for start in range(0, total_frames, self.clip_length):
            end = start + self.clip_length
            clip = frames[start:end]
            # lets also account for inconsistency in length of frames
            if len(clip)<self.clip_length:
            # we pad the clip with the last frame
                pad = [clip[-1]] * (self.clip_length - len(clip))
            # extend clip to reach clip length
                clip.extend(pad)
            clips.append(torch.stack(clip).permute(1,0,2,3))
  
  
        return clips

class ClipLevelDataset(Dataset):
    def __init__(self, video_dataset):
        """
        Args:
            video_dataset: Instance of UCF101ClipDataset
        """
        self.video_dataset = video_dataset
        self.clips = []
        self.labels = []

        # Prepare a flattened list of clips and labels
        for idx in tqdm(range(len(video_dataset))):
            video_clips, video_labels = video_dataset[idx]
            self.clips.extend(video_clips)
            self.labels.extend(video_labels)

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        """
        Returns a single clip and its corresponding label.
        """
        return self.clips[idx], self.labels[idx]


def get_dataloader(data_path, batch_size, clip_length=8,num_clips=1, num_workers=4):
    data_transform = transforms.Compose([
                                               transforms.ToPILImage(),
                                               transforms.Resize((112, 112)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                       ])
    dataset = UCF101ClipDataset(root_dir=data_path, clip_length=clip_length,num_clips=num_clips,transform=data_transform)
    # Wrap it with ClipLevelDataset
    clip_level_dataset = ClipLevelDataset(dataset)
    return DataLoader(clip_level_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)