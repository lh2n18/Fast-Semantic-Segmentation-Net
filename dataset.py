import glob
import os
import numpy as np
import cv2
from torch.utils.data import Dataset 
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as VF

class SegDataset(Dataset):
    def __init__(self, dataRoot, transforms_= None, transforms_mask = None, subFold="train", isTrain=True):

        self.isTrain = isTrain
        if transforms_mask == None:
            self.maskTransform = transforms.Compose([transforms.ToTensor()])
        else:
            self.maskTransform = transforms_mask
        if transforms_== None:
            self.transform = self.maskTransform
        else:
            self.transform = transforms_

        self.imgFiles = sorted(glob.glob(os.path.join(dataRoot, subFold) + "/*.bmp"))
        self.labelFiles = sorted(glob.glob(os.path.join(dataRoot, subFold) + "/*.png"))
        self.len = len(self.imgFiles)

    def __getitem__(self, index):

        idx = index % self.len
        if self.isTrain==True:
            img  = Image.open(self.imgFiles[idx]).convert("RGB")   
            mat = cv2.imread(self.labelFiles[idx], cv2.IMREAD_GRAYSCALE)
            kernel = np.ones((5, 5), np.uint8) # Dilate
            matD = cv2.dilate(mat, kernel)
            mask = Image.fromarray(matD)   # PIL image    

            if np.random.rand(1) > 0.5:
                mask = VF.hflip(mask)
                img  = VF.hflip(img)            
            if np.random.rand(1) > 0.5:
                mask = VF.vflip(mask)
                img = VF.vflip(img)

            img = self.transform(img)
            mask = self.maskTransform(mask)

            return {"img":img, "mask":mask}
        else:
            img = Image.open(self.imgFiles[idx]).convert("RGB")
            img = self.transform(img)
            mat = cv2.imread(self.labelFiles[idx], cv2.IMREAD_GRAYSCALE)
            mask = Image.fromarray(mat)
            mask = self.maskTransform(mask)
            return {"img":img,"mask":mask}

    def __len__(self):
        return len(self.imgFiles)
