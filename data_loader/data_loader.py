import cv2
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader, random_split
import pickle as pkl
import os
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import random
import time
import io
import json
from utils import load_json
from torch.utils.data.distributed import DistributedSampler
from data_loader.sequence_retriver import BaseSequenceRetriever
import torchvision
from collections import defaultdict
from data_loader.image_loader import DefaultImageLoader, NewImageLoader

def create_dataset_from_config(dataset_path, config, batch_size=1, 
                       frame_transform=None, action_transform=None, image_transform=None, 
                       image_size=(224, 224), num_workers=4, view_ids=None, multiview_dir=None,
                       distributed=False, rank=0, world_size=1, gencad=False, 
                       enable_random=False, image_dir=None, sequence_retriever="optimized", sequence_length=10):
    """
    Creates train, validation, and test data loaders with optional distributed training support.
    
    Args:
        dataset_path (str): Path to the dataset
        config (str): Path to the config file
        batch_size (int): Batch size for the data loaders
        frame_transform: Optional transform for frames
        action_transform: Optional transform for actions
        image_transform: Optional transform for CAD images
        image_size (tuple): Size to resize images to
        num_workers (int): Number of workers for data loading
        view_ids (list): List of view IDs to include for each CAD drawing
        multiview_dir (str): Path to directory containing multiview images
        distributed (bool): Whether to use distributed training
        rank (int): Process rank in distributed training
        world_size (int): Total number of processes in distributed training
    
    Returns:
        tuple of dicts: (train_loader, val_loader, test_loader)
        where each dict contains:
            "loader": DataLoader
            "sampler": DistributedSampler or None
    """
    assert os.path.exists(config), f"Config file {config} does not exist"
    config = load_json(config)
    def filter_data(split_type):
        return [name for (name, split) in config.items() if 
                split == split_type]

    train_ids = filter_data("train")
    val_ids = filter_data("val")
    test_ids = filter_data("test")

    def make_dataloader(ids, shuffle=True, enable_random=False, 
                        sequence_retriever="base", split="train"):
        loader, sampler = create_dataloader(dataset_path, 
                                ids=ids, 
                                batch_size=batch_size, 
                                frame_transform=frame_transform, 
                                action_transform=action_transform, 
                                image_transform=image_transform, 
                                image_size=image_size, 
                                num_workers=num_workers,
                                view_ids=view_ids,
                                multiview_dir=multiview_dir,
                                distributed=distributed,
                                rank=rank,
                                world_size=world_size,
                                shuffle=shuffle,
                                gencad=gencad,
                                enable_random=enable_random,
                                image_dir=image_dir,
                                sequence_retriever=sequence_retriever,
                                sequence_length=sequence_length,
                                split=split)
        return {"loader": loader, "sampler": sampler}
    # Create data loaders with appropriate shuffling
    train_loader = make_dataloader(train_ids, shuffle=not distributed, 
    enable_random=enable_random,
    sequence_retriever=sequence_retriever,
    split="train")  # Don't shuffle if distributed
    val_loader = make_dataloader(val_ids, shuffle=False, enable_random=False,
    sequence_retriever=sequence_retriever,
    split="val")  # Never shuffle validation
    test_loader = make_dataloader(test_ids, shuffle=False, enable_random=False,
    sequence_retriever=sequence_retriever,
    split="test")  # Never shuffle test

    return train_loader, val_loader, test_loader


def create_dataloader(dataset_path, batch_size=2, 
                       frame_transform=None, action_transform=None, image_transform=None, 
                       image_size=(224, 224), num_workers=4, seed=42, ids=None, 
                       view_ids=None, multiview_dir=None, distributed=False, 
                       rank=0, world_size=1, shuffle=True, persistent_workers=True, gencad=False, 
                       enable_random=False, image_dir=None, 
                       sequence_retriever="base", sequence_length=10, split="train"):
    """
    Creates a data loader with optional distributed training support.
    
    Args:
        dataset_path (str): Path to the dataset
        batch_size (int): Batch size for the data loader
        frame_transform: Optional transform for frames
        action_transform: Optional transform for actions
        image_transform: Optional transform for CAD images
        image_size (tuple): Size to resize images to
        num_workers (int): Number of workers for data loading
        seed (int): Random seed for reproducibility
        ids (list): List of specific IDs to include in the dataset
        view_ids (list): List of view IDs to include for each CAD drawing
        multiview_dir (str): Path to directory containing multiview images
        distributed (bool): Whether to use distributed training
        rank (int): Process rank in distributed training
        world_size (int): Total number of processes in distributed training
        shuffle (bool): Whether to shuffle the data
        persistent_workers (bool): Whether to use persistent workers
    Returns:
        tuple: (DataLoader, DistributedSampler or None) - Returns both the dataloader and sampler if distributed
    """
    
    assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist"
    if image_dir is not None:
        assert os.path.exists(image_dir), f"Image directory {image_dir} does not exist"
    
    # Create the dataset
    if ids is None:
        dataset = DatasetBase(
                dataset_path=dataset_path,
                frame_transform=frame_transform,
                action_transform=action_transform,
                image_transform=image_transform,
                image_size=image_size,
                view_ids=view_ids,
                multiview_dir=multiview_dir,
                rank=rank,
                gencad=gencad,
                enable_random=enable_random,
                image_dir=image_dir,
                sequence_retriever=sequence_retriever,
                sequence_length=sequence_length,
                split=split
        )
    else:
        dataset = DatasetSpecific(
            dataset_path=dataset_path,
            ids=ids,
            frame_transform=frame_transform,
            action_transform=action_transform,
            image_transform=image_transform,
            image_size=image_size,
            view_ids=view_ids,
            multiview_dir=multiview_dir,
            rank=rank,
            gencad=gencad,
            enable_random=enable_random,
            image_dir=image_dir,
            sequence_retriever=sequence_retriever,
            sequence_length=sequence_length,
            split=split
        )

    # Create sampler for distributed training
    sampler = None
    if distributed:
        if rank == 0:
            print(f"Creating DistributedSampler with world_size: {world_size}, rank: {rank}")
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle
        )
        shuffle = False  # Don't shuffle when using DistributedSampler

    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,  # Enable pin_memory for better performance
        collate_fn=dataset.collate_with_padding,
        drop_last=True,  # Drop last incomplete batch for distributed training
        persistent_workers=persistent_workers
    )
    
    return loader, sampler

def load_retriever(data_files, image_files, sequence_retriever="base", 
                   sequence_length=10, split="train"):
    return BaseSequenceRetriever(data_files, image_files)

class DatasetBase(Dataset):
    """
    dataset_path: path to the state action directory
    image_dir: path to the image directory
    """

    def __init__(self, dataset_path, 
                 frame_transform=None,
                 action_transform=None,
                 image_transform=None,
                 image_size=(224, 224),
                 view_ids=None,
                 multiview_dir=None,
                 rank=0,
                 gencad=False, 
                 enable_random=False,
                 image_dir=None,
                 sequence_retriever="base",
                 sequence_length=10,
                 split="train"):
        self.dataset_path = dataset_path
        self.image_dir = image_dir
        if self.image_dir is None:
            raise ValueError("Image directory is None")
        if self.image_dir == self.dataset_path:
            self.image_loader = DefaultImageLoader(image_dir)
        else:
            self.image_loader = NewImageLoader(image_dir, enable_random)

        
        self.frame_transform = frame_transform
        self.action_transform = action_transform
        self.image_transform = image_transform
        self.gencad = gencad
        self.image_size = image_size
        self.dataset_type = dataset_path
        self.view_ids = view_ids if view_ids is not None else []
        self.multiview_dir = multiview_dir
        self.rank = rank
        self.data_files, self.image_files = self.create_data_and_image_files()
        self.sequence_retriever = load_retriever(
            self.data_files, self.image_files, sequence_retriever, sequence_length, split)
        # Check if all samples have the requested multiview images
        if self.view_ids and self.rank == 0:
            print("Checking multiview availability")
            self.check_multiview_availability()
        if self.rank == 0:
            print("Checking data files")
            # self.validate_data_files()
            print("Dataset is valid")

    def validate_data_files(self):
        for i in range(self.__len__()):
            sample = self.__getitem__(i)
            actions = sample['actions']
            # Assert that action class labels are between 0 and 4 inclusive
            assert (actions[:, 0] >= 0).all() and (actions[:, 0] <= 4).all(), \
                f"Action class labels must be between 0 and 4, but got values outside this range"
            # Check that remaining action values are between -1 and 999
            action_values = actions[:, 1:]  # All columns except class label
            assert (action_values >= -1).all() and (action_values <= 999).all(), \
                f"Action values must be between -1 and 999, but got values outside this range"

            
    def check_multiview_availability(self):
        """Check if all samples in the dataset contain the requested multiview images."""
        missing_views = {}
        for i, image_file in enumerate(self.image_files):
            base_file_id = os.path.basename(image_file).split("_")[0]
            
            # Determine directory for multiview images
            if self.multiview_dir:
                base_dir = self.multiview_dir
            else:
                base_dir = os.path.dirname(image_file)
            
            for view_id in self.view_ids:
                view_path = os.path.join(base_dir, base_file_id[:4], f"{base_file_id}_{view_id}.png")
                if not os.path.exists(view_path):
                    if base_file_id not in missing_views:
                        missing_views[base_file_id] = []
                    missing_views[base_file_id].append(view_id)
        
        if missing_views:
            print(f"Warning: {len(missing_views)} samples are missing some requested view IDs:")
            for file_id, views in list(missing_views.items())[:5]:  # Show first 5 examples
                print(f"  - Sample {file_id} is missing views: {', '.join(views)}")
            if len(missing_views) > 5:
                print(f"  ... and {len(missing_views) - 5} more samples with missing views")
            raise ValueError(f"Dataset is missing requested multiview images for {len(missing_views)} samples.")
        elif self.rank == 0:
            print(f"All {len(self.image_files)} samples have all requested multiview images.")

    def create_data_and_image_files(self):
        data_files = []
        image_files = []
        data_ids = set()
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('_data.pkl'):
                    data_files.append(os.path.join(root, file))
                if file.endswith('.png'):
                    data_ids.add(file.split("_")[0])
        image_files = list(data_ids)
        data_files.sort()
        image_files.sort()
        assert len(data_files) == len(image_files), "Number of data files and image files must be the same"
        return data_files, image_files

    def pad_array(self, max_len, array):
        pad_size = max_len - array.shape[0]
        if pad_size > 0:
            padding = torch.full((pad_size, *array.shape[1:]), fill_value=-1, dtype=array.dtype)
            array = torch.cat([array, padding], dim=0)
        return array
    

    def collate_with_padding(self, batch):
        # Find max sequence length in this batch
        max_len = max(item['frames'].shape[0] for item in batch)
        
        # Pad each sequence to max_len
        padded_batch = []
        for item in batch:
            frames = item['frames']
            actions = item['actions']
            timesteps = item['timesteps']
            
            # Use pad_array helper for all tensors
            frames = self.pad_array(max_len, frames)
            actions = self.pad_array(max_len, actions)
            timesteps = torch.arange(max_len)  # Just create new timesteps up to max_len
            
            padded_item = {
                'frames': frames,
                'actions': actions,
                'cad_image': item['cad_image'],
                'timesteps': timesteps
            }
            
            # Add multiview_images if they exist
            if 'multiview_images' in item and item['multiview_images'] is not None:
                padded_item['multiview_images'] = item['multiview_images']
            else:
                padded_item['multiview_images'] = None
                
            padded_batch.append(padded_item)
            
        # Stack all items in batch
        result = {
            'frames': torch.stack([item['frames'] for item in padded_batch]),
            'actions': torch.stack([item['actions'] for item in padded_batch]),
            'cad_image': torch.stack([item['cad_image'] for item in padded_batch]),
            'timesteps': torch.stack([item['timesteps'] for item in padded_batch])
        }
        
        # Handle multiview images - only stack if they exist for all items
        if all(item['multiview_images'] is not None for item in padded_batch):
            result['multiview_images'] = torch.stack([item['multiview_images'] for item in padded_batch])
        else:
            result['multiview_images'] = None
            
        return result
    
    def resize_frames(self, frames):
        # Normalize frames to [0,1] range
        frames = frames.astype(np.float32) / 255.0
        frames = np.stack(frames)  # [N, H, W, C]
        
        # Transpose to [N, C, H, W] to match PyTorch's expected format
        frames = frames.transpose(0, 3, 1, 2)
        
        frames_tensor = torch.from_numpy(frames)
        return frames_tensor
    
    def resize_frames_slow(self, frames):
        processed_frames = []
        for frame in frames:
            # Convert to PIL for better resizing
            pil_frame = Image.fromarray(frame)
            resized_frame = pil_frame.resize(self.image_size, Image.Resampling.BILINEAR)
            # Convert back to numpy and normalize
            frame_array = np.array(resized_frame).astype(np.float32) / 255.0
            frame_tensor = torch.from_numpy(frame_array).permute(2, 0, 1)
            processed_frames.append(frame_tensor)
        frames_tensor = torch.stack(processed_frames) 
        return frames_tensor
    
    def _load_image_cv2(self, image_path):
        cad_image = cv2.imread(image_path)
        cad_image = cv2.cvtColor(cad_image, cv2.COLOR_BGR2RGB)
        cad_image = cv2.resize(cad_image, self.image_size)
        cad_image = cad_image.astype(np.float32) / 255.0
        cad_image = torch.from_numpy(cad_image).permute(2, 0, 1)
        return cad_image
    

    
    def _get_item(self, idx):
        """ Given an idx return cad_image, frames and actions """
        frames, actions, base_file_id = self.sequence_retriever.get_sequence(idx)
        # data = pickle.load(open(self.data_files[idx], 'rb'))
        # frames = data['frames']
        # actions = data['actions']
        
        # Get the base file ID and path
        # base_file_id = os.path.basename(self.image_files[idx]).split("_")[0]
        base_file_id = os.path.basename(self.image_files[idx]).split("_")[0]
        # Load regular CAD image 
        cad_image = self.image_loader.get_image(base_file_id)
        
        # Handle multi-view images separately
        multiview_images = None
        if self.view_ids:
            # Determine directory for multiview images
            if self.multiview_dir:
                base_dir = self.multiview_dir
                
            views = []
            for view_id in self.view_ids:
                view_path = os.path.join(base_dir, base_file_id[:4], f"{base_file_id}_{view_id}.png")
                if not os.path.exists(view_path):
                    raise ValueError(f"Missing view {view_id} for file {base_file_id}")
                view_img = cv2.imread(view_path)
                views.append(view_img)
            multiview_images = views
            
        return actions, frames, cad_image, multiview_images

    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data_files):
            raise IndexError('Index out of range')
        
        actions, frames, cad_image, multiview_images = self._get_item(idx)
        
        # Process frames
        if self.frame_transform:
            processed_frames = []
            for frame in frames:
                pil_frame = Image.fromarray(frame)
                processed_frame = self.frame_transform(pil_frame)
                processed_frames.append(processed_frame)
            frames_tensor = torch.stack(processed_frames)
        else:
            frames_tensor = self.resize_frames_slow(frames)
        
        actions_tensor = torch.from_numpy(actions.astype(np.float32))

        # Process main CAD image
        if self.gencad:
            preprocess = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(256),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                                    std=[0.5, 0.5, 0.5]),
                            ])
            cad_image = cv2.cvtColor(cad_image, cv2.COLOR_BGR2RGB)
            img = np.array(cad_image)
            l, h = 100, 200
            img = cv2.Canny(img, l, h)
            img = img[:, :, None]
            img = np.concatenate([img, img, img], axis=2)
            img = Image.fromarray(img)
            cad_image = preprocess(img)
        else:
            cad_image = cv2.cvtColor(cad_image, cv2.COLOR_BGR2GRAY)
            cad_image = cv2.resize(cad_image, self.image_size)
            cad_image = cad_image.astype(np.float32) / 255.0
            cad_image = torch.from_numpy(cad_image).unsqueeze(0)  # Add channel dimension
            if self.image_transform:
                cad_image = self.image_transform(cad_image)

        # Process multiview images if they exist
        multiview_tensor = None
        if multiview_images is not None:
            processed_views = []
            for view in multiview_images:
                view = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
                view = cv2.resize(view, self.image_size)
                view = view.astype(np.float32) / 255.0
                view = torch.from_numpy(view).unsqueeze(0)
                if self.image_transform:
                    view = self.image_transform(view)
                processed_views.append(view)
            multiview_tensor = torch.stack(processed_views)

        # Generate timesteps
        timesteps = torch.arange(frames_tensor.shape[0])
        
        
        # Print timing breakdown
        # print(f"Timing breakdown - Total: {total_time:.3f}s "
        #       f"(Load: {load_time:.3f}s ({load_time/total_time*100:.1f}%), "
        #       f"Frames: {frame_time:.3f}s ({frame_time/total_time*100:.1f}%), "
        #       f"CAD: {cad_time:.3f}s ({cad_time/total_time*100:.1f}%))")

        return {
            'frames': frames_tensor,
            'actions': actions_tensor,
            'cad_image': cad_image,
            'multiview_images': multiview_tensor,
            'timesteps': timesteps
        }

    def __len__(self):
        return len(self.data_files)
    
    

class DatasetSpecific(DatasetBase):
    """
    Dataset class that contains samples only from ids
    """
    def __init__(self, dataset_path, 
                 ids,
                 frame_transform=None,
                 action_transform=None,
                 image_transform=None,
                 image_size=(224, 224),
                 view_ids=None,
                 multiview_dir=None,
                 rank=0,
                 gencad=False,
                 enable_random=False,
                 image_dir=None,
                 sequence_retriever="base",
                 sequence_length=10,
                 split="train"):
        self.ids = set(ids)
        super().__init__(
            dataset_path=dataset_path,
            frame_transform=frame_transform,
            action_transform=action_transform,
            image_transform=image_transform,
            image_size=image_size,
            view_ids=view_ids,
            multiview_dir=multiview_dir,
            rank=rank,
            gencad=gencad,
            enable_random=enable_random,
            image_dir=image_dir,
            sequence_retriever=sequence_retriever,
            sequence_length=sequence_length,
            split=split
        )

    def create_data_and_image_files(self):
        data_files = []
        image_files = []
        data_ids = set()
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                file_id = file.split("_")[0]
                if file_id not in self.ids:
                    continue
                if file.endswith('_data.pkl'):
                    data_files.append(os.path.join(root, file))
                if file.endswith('.png'):
                    image_files.append(os.path.join(root, file))
                    data_ids.add(file_id)
        data_ids = list(data_ids)
        data_files.sort()
        image_files.sort()
        assert len(data_files) == len(image_files), f"Number of data files and image files must be the same. len(data_files): {len(data_files)}, len(image_files): {len(image_files)}"
        return data_files, image_files

