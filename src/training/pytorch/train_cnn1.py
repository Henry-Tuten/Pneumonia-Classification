import torch
import torch.nn as nn
import transformers
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#------------------Hyper Parameters----------------------
batch_size = 32
learning_rate = 3e-4
dropout = 0.2

#--------------------------------------------------------

#------------------Check Cuda----------------------------

if torch.cuda.is_available():
    print("CUDA is available!")
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

#--------------------------------------------------------

#------------------Data_Loader---------------------------

train_dataset = datasets.Imagefolder(root= "data/train/", transform=transform)

dataloader= DataLoader(train_dataset, batch_size, shuffle=True)

#--------------------------------------------------------

#------------------Instantiate Model---------------------

model = pyt_transf()

#--------------------------------------------------------