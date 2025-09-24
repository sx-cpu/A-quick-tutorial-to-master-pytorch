# Try to use pretrained model

import os
import numpy as np
import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable
import torch.cuda
import torchvision.transforms as transforms
from PIL import Image


# from IPython.display import Image, display
# display(Image(filename=path))

path = "AI_security_example/picture/pig.png"

# preprocess
resnet50 = models.resnet50(pretrained=True).eval()
img = Image.open(path)
img = img.resize((224, 224))
img = np.array(img).copy().astype(np.float32)

# mean and standardise
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
img /= 255.0
img = (img - mean) / std
img = img.transpose(2, 0, 1)
img = np.expand_dims(img, axis=0)
img = Variable(torch.from_numpy(img).float())

label = np.argmax(resnet50(img).data.cpu().numpy())
print(f'label={label}')
# label = 341

