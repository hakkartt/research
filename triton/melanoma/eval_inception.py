import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms, models
from torch import nn
import torch.nn.functional as F
import sklearn.metrics as metrics
from cleverhans.torch.attacks.fast_gradient_method import (
        fast_gradient_method as FGM
        )
from cleverhans.torch.attacks.projected_gradient_descent import (
        projected_gradient_descent as PGD
        )
from cleverhans.torch.attacks.carlini_wagner_l2 import (
        carlini_wagner_l2 as CW
        )
from cleverhans.torch.attacks.spsa import (
        spsa as SPSA
        )
import time
    
fpath = "/scratch/work/hakkina7/melanoma/"
print("File path is {}".format(fpath))

# Set the device
dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)
print("Device is {}".format(device))

# Set file name
fname = 'inceptionV3.pth'

h, w = (299, 299)
# Get proper dataloader with normalized images
transformations = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor([0.0, 0.0, 0.0]), torch.Tensor([1.0, 1.0, 1.0]))
])

testset = torchvision.datasets.ImageFolder(root=fpath+"data/test/",
                                           transform=transformations)
testloader = torch.utils.data.DataLoader(dataset=testset,
                                         batch_size=5,
                                         shuffle=False)

def evaluate(model, testloader, device=torch.device(dev),
             adv='No attack', n_iter=0):
    """Evaluate model performance on test set"""
    
    model.eval()
    correct = 0
    total = 0
    true = torch.zeros(0, dtype=torch.long, device='cpu')
    preds = torch.zeros(0, dtype=torch.long, device='cpu')
    probs = torch.zeros(0, dtype=torch.float, device='cpu')
    __str__ = adv
    start = time.time()
    
    for i, (images, labels) in enumerate(testloader):
        images = images.to(device)
        if adv == 'FGM':
            images = FGM(model, images, eps=0.02, norm=np.inf)
        elif adv == 'PGD':
            if n_iter > 0:
                images = PGD(model, images, eps=0.02, eps_iter=0.01,
                             nb_iter=n_iter, norm=np.inf)
                __str__ = adv + ' with {} iterations'.format(n_iter)
            else:
                print("Need proper input for n_iter.")
            
        elif adv == 'CW':
            images = CW(model, images, n_classes=2)
        elif adv == 'SPSA':
            if n_iter > 0:
                images = SPSA(model, images, eps=0.02,
                              nb_iter=n_iter, norm=np.inf,
                              sanity_checks=False)
                __str__ = adv + ' with {} iterations'.format(n_iter)
            else:
                print("Need proper input for n_iter.")
        outputs = model(images).detach().cpu()
        images = images.detach().cpu()
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        if adv == 'No attack':
            true = torch.cat([true, labels.view(-1)])
            preds = torch.cat([preds, pred.view(-1)])
            prob = (F.softmax(outputs, dim=1))[:, 1]
            probs = torch.cat([probs, prob.view(-1)])
    
    t = (time.time()-start) // 60
    print()
    print("##################################################################")
    print(__str__)
    print('Computing took ~{:.0f} minutes.'.format(t))
    acc = 100 * correct/total
    print("\nClassified {:.2f} % of test images correctly.".format(acc))
    print("##################################################################")
    print()
    
    if adv == 'No attack':    
        print("Precision on test set: {}".format(
            np.round(metrics.precision_score(true.numpy(), preds.numpy(), average=None), 2)
        ))
        print("Recall on test set: {}".format(
            np.round(metrics.recall_score(true.numpy(), preds.numpy(), average=None), 2)
        ))
        print("F1-score on test set: {}".format(
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
        plt.savefig(fpath+'roc_{}.png'.format(fname))
        plt.show()
        print("ROC AUC on test set: {}".format(aucs))
        print("Confusion matrix on test set:")
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
model = models.inception_v3(pretrained=False)
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

# Evaluate the model performance on test set
print("Evaluating the {} model on the test set.".format(fname))
evaluate(model, testloader)
evaluate(model, testloader, adv='FGM')
iters = [1, 2, 5, 10, 20]
for i in iters:
    evaluate(model, testloader, adv='PGD', n_iter=i)
evaluate(model, testloader, adv='CW')
for i in iters:
    evaluate(model, testloader, adv='SPSA', n_iter=i)


