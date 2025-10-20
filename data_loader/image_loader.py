import os
import cv2
import random
from collections import defaultdict


class ImageLoader:

    def __init__(self, image_dir, randomize_images=False):
        self.image_dir = image_dir
        self.randomize_images = randomize_images

    def __len__(self):
        return len(os.listdir(self.image_dir))
    
    def get_image_id(self, image_id):
        return f"{image_id[:4]}/{image_id}"
    
    def check_exists(self, image_id):
        id_path = self.get_image_path(image_id)
        return os.path.exists(id_path)
    
    def get_image_path(self, image_id):
        raise NotImplementedError("Subclasses must implement this method")
    
    
    def _get_item(self, image_id):
        raise NotImplementedError("Subclasses must implement this method")
    

class DefaultImageLoader(ImageLoader):
    """ Image loader with previous directory structure (all data in one directory) """
    def __init__(self, image_dir):
        self.image_dir = image_dir
        super().__init__(image_dir, randomize_images=False)

    def get_image_path(self, image_id):
        id_path = os.path.join(self.image_dir, self.get_image_id(image_id))
        id_path = f"{id_path}_frame.png"
        return id_path
    
    def get_image(self, image_id):
        id_path = self.get_image_path(image_id)
        return cv2.imread(id_path)
    
class NewImageLoader(ImageLoader):
    """ Image loader with new directory structure (cad image in one directory, frames in another) """
    def __init__(self, image_dir, enable_random=False):
        self.image_dir = image_dir
        super().__init__(image_dir, randomize_images=enable_random)
        image_mapping = defaultdict(list)
        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                if file.endswith('.png'):
                    image_mapping[file.split('_')[0]].append(os.path.join(root, file))
        self.image_mapping = image_mapping

    def get_image_path(self, image_id):
        id_base = os.path.join(self.image_dir, self.get_image_id(image_id))
        if self.randomize_images:
            samples = self.image_mapping[image_id]
            return random.choice(samples)
        else:
            id_path = f"{id_base}_0.png"
        return id_path

    def get_image(self, image_id):
        id_path = self.get_image_path(image_id)
        return cv2.imread(id_path)
