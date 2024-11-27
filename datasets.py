import math
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
import os
from tqdm import tqdm

# supress warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Subcarrier selection for 3DO dataset
# 52 L-LTF subcarriers 
csi_vaid_subcarrier_index = []
csi_vaid_subcarrier_index += [i for i in range(6, 32)]
csi_vaid_subcarrier_index += [i for i in range(33, 59)]
# 56 HT-LTF: subcarriers 
#csi_vaid_subcarrier_index += [i for i in range(66, 94)]     
#csi_vaid_subcarrier_index += [i for i in range(95, 123)] 
CSI_SUBCARRIERS = len(csi_vaid_subcarrier_index) 

# 3DO dataset class
class TDODataset(Dataset):
    def __init__(self, dataPath,opt):
        self.dataPath = dataPath
        folderEndIndex = dataPath.rfind('/')
        self.dataFolder = self.dataPath[0:folderEndIndex]
        self.imagePath = os.path.dirname(dataPath)
        self.windowSize = opt.ws
        assert self.windowSize % 2 == 1
        self.windowSizeH = math.ceil(self.windowSize/2)

        # read data from .csv file
        data = pd.read_csv(dataPath)
        csi = data['data']
        self.x = data['x']
        self.y = data['y']
        self.z = data['z']
        self.c = data['class']

        # pre-compute or load complex CSI cache
        if os.path.exists(self.dataFolder + f"/csiposreg_complex.npy"):
            csiComplex = np.load(self.dataFolder + f"/csiposreg_complex.npy")
        else:
            csiComplex = np.zeros(
                [len(csi), CSI_SUBCARRIERS], dtype=np.complex64)
            for s in tqdm(range(len(csi))):
                for i in range(CSI_SUBCARRIERS):
                    sample = csi[s][1:-1].split(',')
                    sample = np.array([int(x) for x in sample])
                    csiComplex[s][i] = complex(sample[csi_vaid_subcarrier_index[i] * 2], sample[csi_vaid_subcarrier_index[i] * 2 - 1])
            np.save(self.dataFolder + f"/csiposreg_complex.npy", csiComplex)

        self.features = np.abs(csiComplex) # amplitude
        #self.features = np.angle(csiComplex) # phase

        # number of samples
        self.dataSize = len(self.features)-self.windowSize 

    def __len__(self):
        return self.dataSize
    
    def __getitem__(self, index):
        index = index + self.windowSizeH # add index offset to avoid border regions
        location = np.array([self.x[index], self.y[index], self.z[index]]) # get 3D location label
        c = self.c[index]-1 # get class label, activity labels start with 1 (label 0 is reserved for background)
        featureWindow = self.features[index-self.windowSizeH:index+self.windowSizeH-1] # get feature window
        featureWindow = np.transpose(featureWindow) # transpose to [self.windowSize, CSI_SUBCARRIERS]
        featureWindow = np.expand_dims(featureWindow, axis=0) # add channel dimension
        
        # convert to tensor
        featureWindow = torch.tensor(featureWindow)

        return featureWindow, location, c 

