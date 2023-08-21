from torch.utils.data import Dataset
import json
import numpy as np
import os
import cv2

class MyDataSet(Dataset):
    """自定义数据集"""

    CLASSES = []
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id.split('.')[0] + '.json') for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        num_classes=len(self.class_values)
        with open (self.masks_fps[i],"r")as f:
            mask_json=json.load(f)
            shapes=mask_json.get("shapes",[])
            masks =np.zeros((num_classes,image.shape[0],image.shape[1]),dtype=np.uint8)

            for shape in shapes:
                points=np.array(shape["points"],dtype=np.int32)
                
                label=shape["label"].lower()
                label_index=self.CLASSES.index(label)
                color=(255,255,255)
    
                cv2.fillPoly(masks[label_index],[points],color) 
            mask=np.stack(masks,axis=-1).astype("float")       
        
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
