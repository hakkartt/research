import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms, models
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
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


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class VIB(nn.Module):
    def __init__(self,
                 input_size=2048,
                 K=256,
                 num_latent=12,
                 output_size=2,
                 beta=0.01):
        super(VIB, self).__init__()
        self.input_size = input_size
        self.K = K
        self.output_size = output_size
        self.beta = beta
        
        # binary classification (cross entropy loss)
        self.fit_loss = nn.CrossEntropyLoss()
        
        # encoder layers
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.encoder_mean = nn.Linear(1024, K)
        self.encoder_std = nn.Linear(1024, K)
        # encoder initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.xavier_uniform_(self.encoder_mean.weight)
        nn.init.constant_(self.encoder_mean.bias, 0.0)
        nn.init.xavier_uniform_(self.encoder_std.weight)
        nn.init.constant_(self.encoder_std.bias, 0.0)
        
        # decoder layer
        self.decoder = nn.Linear(K, output_size)
        # decoder initialization
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.constant_(self.decoder.bias, 0.0)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.encoder_mean(x)
        sigma = F.softplus(self.encoder_std(x)-5)
        latent = Normal(mu, sigma).sample()
        outputs = self.decoder(latent)
        probs = (F.softmax(outputs, dim=1))[:, 1]
        return outputs, probs, latent, mu, sigma
    
    def compute_loss(self, outputs, labels, latent, mu, sigma, device):
        # compute KL loss between encoder output parameters and prior
        Q = Normal(mu, sigma)
        P = Normal(
                torch.zeros(1, self.K).to(device), 
                torch.ones(1, self.K).to(device)
        )
        KL_loss = kl_divergence(Q, P).mean()
        fit_loss = self.fit_loss(outputs, labels)
        # total_loss = fit_loss + beta * KL_loss (aka complexity_loss)
        loss = fit_loss + self.beta * KL_loss
        return loss, fit_loss, KL_loss
    

def img_difference(adv, real):
    """Calculate the average pixel difference between two images."""
    return torch.mean(torch.abs(real.view(1, -1) - adv.view(1, -1))).item()


def evaluate(model, testloader, device, fname,
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
                __str__ = adv + '_with_{}_iterations_and_{}_eps'.format(n_iter, eps)
                diff = img_difference(adv_images, images) 
            else:
                print("Need proper input for n_iter.")
        else:
            adv_images = images
            diff = 0
        if diff > max_diff:
            max_diff = diff
            to_be_plotted = [images.detach().cpu(), adv_images.detach().cpu()]
        outputs, prob, _, _, _ = model(adv_images)
        outputs, prob = outputs.detach().cpu(), prob.detach().cpu()
        images = images.detach().cpu()
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        if adv == 'No attack':
            true = torch.cat([true, labels.view(-1)])
            preds = torch.cat([preds, pred.view(-1)])
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
        plt.show()
    else:
        grid = torchvision.utils.make_grid(torch.cat(to_be_plotted, axis=0),
                                           nrow=len(images))
        plt.figure(figsize=(20, 40))
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.savefig(fpath+'comparison_{}_{}.png'.format(fname[:-4], __str__))

fpath = "/scratch/work/hakkina7/chest_xray/"
print("File path is {}".format(fpath))

# Set the device
dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)
print("Device is {}".format(device))

# Set file name
fname_base = 'resnet50_nonweighted.pth'
fname_vib = 'resnet50_vib_nw.pth'

h, w = (224, 224)
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
model = models.resnet50(pretrained=False)
n_labels = 2
n_inputs = model.fc.in_features
model.fc = nn.Linear(n_inputs, n_labels)
# Load model params
if torch.cuda.is_available():
    model.load_state_dict(torch.load(fpath+fname_base))
    model.to(device)
else:
    model.load_state_dict(torch.load(fpath+fname_base,
                                     map_location=torch.device('cpu')))
print("Pre-trained base model succesfully loaded from {}".format(fpath+fname_base))

# Initialize Variational Information Bottleneck
vib = VIB()
# Load VIB params
if torch.cuda.is_available():
    vib.load_state_dict(torch.load(fpath+fname_vib))
    vib.to(device)
else:
    vib.load_state_dict(torch.load(fpath+fname_vib,
                                     map_location=torch.device('cpu')))
print("Pretrained VIB succesfully loaded from {}".format(fpath+fname_vib))

# Replace last layer of the base model with VIB
model.fc = vib


# Evaluate the model performance on test set
iters = [1, 2, 5, 10, 20]
epss = [0.02, 0.05, 0.1, 0.2, 0.5, 1]
print("Evaluating the {} model on the test set.".format(fname_vib))
evaluate(model, testloader, device, fname_vib)
evaluate(model, testloader, device, fname_vib, adv='FGM')
evaluate(model, testloader, device, fname_vib, adv='CW')
for i in iters:
    evaluate(model, testloader, device, fname_vib, adv='PGD', n_iter=i)
for i in iters:
    for e in epss:
        evaluate(model, testloader, device,
                 fname_vib, adv='SPSA',
                 n_iter=i, eps=e)
