#!/usr/bin/env python
# coding: utf-8

# # Melanoma, Fundoscopy - ResNet50
# -------------------

# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import torch
import torchvision
from torchvision import transforms, models
from torch import nn, optim
import torch.nn.functional as F
import sklearn.metrics as metrics
   
fpath = "/scratch/work/hakkina7/melanoma/"
print("File path is {}".format(fpath))

# Set the device
dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)
print("Device is {}".format(device))

# Set file name
fname = 'resnet50.pth'


# ## Data preprocessing
# ---------

# In[24]:


# Takes a while to run the mean and std calculation, so save the values here
# NB: These have to be calculated again if the image size is changed
# NB: Modifying the batch size doesn't matter, mean and std stay the same
# mean = [0.6678, 0.5298, 0.5245] # h, w = (224, 224)
# std = [0.1331, 0.1473, 0.1587] # h, w = (224, 224)

h, w = (224, 224)
# Get proper dataloader with normalized images
transformations = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor([0.0, 0.0, 0.0]), torch.Tensor([1.0, 1.0, 1.0]))
])

trainset = torchvision.datasets.ImageFolder(root=fpath+"data/train/",
                                           transform=transformations)
testset = torchvision.datasets.ImageFolder(root=fpath+"data/test/",
                                           transform=transformations)
valset = torchvision.datasets.ImageFolder(root=fpath+"data/val/",
                                           transform=transformations)

trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=8, shuffle=False)
valloader = torch.utils.data.DataLoader(dataset=valset, batch_size=8, shuffle=False)


# ## Evaluate
# -----------

# In[30]:


def evaluate(model, valloader):
    """Evaluate model performance on validation set"""
    
    model.eval()
    correct = 0
    total = 0
    true = torch.zeros(0, dtype=torch.long, device='cpu')
    preds = torch.zeros(0, dtype=torch.long, device='cpu')
    probs = torch.zeros(0, dtype=torch.float, device='cpu')
    
    with torch.no_grad():
    
        for i, (images, labels) in enumerate(valloader):
            images, labels = images.cpu(), labels.cpu()
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            true = torch.cat([true, labels.view(-1)])
            preds = torch.cat([preds, pred.view(-1)])
            prob, _ = (F.softmax(outputs, dim=1)).topk(1, dim=1)
            probs = torch.cat([probs, prob.view(-1)])
    
    acc = 100 * correct/total
    print("Classified {:.2f} % of validation images correctly.".format(
        acc
    ))
    print("Precision on validation set: {}".format(
        np.round(metrics.precision_score(true.numpy(), preds.numpy(), average=None), 2)
    ))
    print("Recall on validation set: {}".format(
        np.round(metrics.recall_score(true.numpy(), preds.numpy(), average=None), 2)
    ))
    print("F1-score on validation set: {}".format(
        np.round(metrics.f1_score(true.numpy(), preds.numpy(), average=None), 2)
    ))
    classes = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
    aucs = {}
    for i, c in enumerate(classes):
        fpr, tpr, _ = metrics.roc_curve(true.numpy(),
                                        probs.numpy(), 
                                        pos_label=i)
        aucs[c] = np.round(metrics.auc(fpr, tpr), 2)
        plt.plot(fpr, tpr, label='{} AUC = {:.2f}'.format(c, metrics.auc(fpr, tpr)))
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(fpath+'roc_resnet.png')
    plt.show()
    print("ROC AUC on validation set: {}".format(aucs))
    print("Confusion matrix on validation set:")
    print(pd.DataFrame(metrics.confusion_matrix(true.numpy(), preds.numpy()), 
                       columns=['pred as AK', 'pred as BCC', 
                                'pred as BKL', 'pred as DF',
                                'pred as MEL', 'pred as NV',
                                'pred as SCC', 'pred as VASC'], 
                       index=['true class AK', 'true class BCC',
                              'true class BKL', 'true class DF',
                              'true class MEL', 'true class NV',
                              'true class SCC', 'true class VASC']))
    print()
    

# Initialize model
model = models.resnet50(pretrained=False)
n_labels = 8
n_inputs = model.fc.in_features
model.fc = nn.Linear(n_inputs, n_labels)

# Load model params
if torch.cuda.is_available():
    model.load_state_dict(torch.load(fpath+fname))
    model.to(device)
else:
    model.load_state_dict(torch.load(fpath+fname,
                                     map_location=torch.device('cpu')))
    
print("Model succesfully loaded from {}".format(fpath+fname))

# Evaluate the model performance on validation set
print("Evaluating the ResNet50 model on the validation set.")
evaluate(model, valloader)


