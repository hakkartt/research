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


def img_difference(adv, real):
    """Calculate the average pixel difference between two images."""
    return torch.mean(torch.abs(real.view(1, -1) - adv.view(1, -1))).item()


def evaluate(model, testloader, device,
             adv='No attack', n_iter=0, eps=0.02):
    """Evaluate model performance on test set"""
    
    model.eval()
    correct = 0
    total = 0
    true = torch.zeros(0, dtype=torch.long, device='cpu')
    preds = torch.zeros(0, dtype=torch.long, device='cpu')
    probs = torch.zeros(0, dtype=torch.float, device='cpu')
    __str__ = adv
    start = time.time()
    
    max_diff = 0
    
    for i, (images, labels) in enumerate(testloader):
        images = images.to(device)
        if adv == 'FGM':
            adv_images = FGM(model, images, eps=eps, norm=np.inf)
            diff = img_difference(adv_images, images) 
        elif adv == 'PGD':
            if n_iter > 0:
                adv_images = PGD(model, images, eps=eps, eps_iter=0.01,
                             nb_iter=n_iter, norm=np.inf)
                __str__ = adv + '_with_{}_iterations_and_{}_eps'.format(n_iter, eps)
                diff = img_difference(adv_images, images) 
            else:
                print("Need proper input for n_iter.")
            
        elif adv == 'CW':
            adv_images = CW(model, images, n_classes=2)
            diff = img_difference(adv_images, images) 
        elif adv == 'SPSA':
            if n_iter > 0:
                adv_images = SPSA(model, images, eps=eps,
                                  nb_iter=n_iter, norm=np.inf,
                                  sanity_checks=False)
                __str__ = adv + '_with_{}_iterations'.format(n_iter)
                diff = img_difference(adv_images, images) 
            else:
                print("Need proper input for n_iter.")
        else:
            adv_images = images
            diff = 0
        if diff > max_diff:
            max_diff = diff
            to_be_plotted = [images.detach().cpu(), adv_images.detach().cpu()]
        outputs = model(adv_images).detach().cpu()
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
        print("Precision on test set: {:.2f}".format(
            metrics.precision_score(true.numpy(), preds.numpy())
        ))
        print("Recall on test set: {:.2f}".format(
            metrics.recall_score(true.numpy(), preds.numpy())
        ))
        print("F1-score on test set: {:.2f}".format(
            metrics.f1_score(true.numpy(), preds.numpy())
        ))
        fpr, tpr, _ = metrics.roc_curve(true.numpy(), probs.numpy(), pos_label=1)
        print("AUC score on test set: {:.2f}".format(
            metrics.auc(fpr, tpr)
        ))
        print("Confusion matrix on test set:")
        print(pd.DataFrame(metrics.confusion_matrix(true.numpy(), preds.numpy()), 
                           columns=['pred as NEG', 'pred as POS'], 
                           index=['true class NEG', 'true class POS']))
        print()
        
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % metrics.auc(fpr, tpr))
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(fpath+'roc_{}.png'.format(fname[:-4]))
        plt.close()
    else:
        grid=torchvision.utils.make_grid(torch.cat(to_be_plotted, axis=0),
                                         nrow=len(images))
        plt.figure(figsize=(20, 40))
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.savefig(fpath+'comparison_{}_{}.png'.format(fname[:-4], __str__))
        plt.close()


###############################################################################
# MAIN PROGRAM
###############################################################################

# Set path
fpath = "/scratch/work/hakkina7/chest_xray/"
print("File path is {}".format(fpath))

# Set file name
fname = 'inceptionV3_nonweighted.pth'

# Set the device
dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)
print("Device is {}".format(device))

h, w = (299, 299)
# Get proper dataloader with normalized images
transformations = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor([0.0, 0.0, 0.0]),
                         torch.Tensor([1.0, 1.0, 1.0]))
])

testset = torchvision.datasets.ImageFolder(root=fpath+"data/test/",
                                           transform=transformations)
testloader = torch.utils.data.DataLoader(dataset=testset,
                                         batch_size=5,
                                         shuffle=False)

# Initialize model
model = models.inception_v3(pretrained=False)
n_labels = 2
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
iters = [1, 2, 5, 10, 20]
epss = [0.02, 0.05, 0.1, 0.2, 0.5, 1]
print("Evaluating the {} model on the test set.".format(fname))
evaluate(model, testloader, device)
evaluate(model, testloader, device, adv='FGM')
evaluate(model, testloader, device, adv='CW')
for i in iters:
    evaluate(model, testloader, device, adv='PGD', n_iter=i)
for i in iters:
    for e in epss:
        evaluate(model, testloader, device, adv='SPSA', n_iter=i, eps=e)
