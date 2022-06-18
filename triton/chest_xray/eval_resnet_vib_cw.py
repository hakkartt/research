import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms, models
from torch import nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
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
        self.num_latent = num_latent
        self.output_size = output_size
        self.beta = beta
        
        # weighted CE loss as classification loss
        self.n_samples = [1266, 3418]
        self.weights = torch.FloatTensor(
                [1 / (x / sum(self.n_samples)) for x in self.n_samples]
                ).to(device)
        self.fit_loss = nn.CrossEntropyLoss(weight=self.weights)
        
        self.encoder_mean = nn.Linear(input_size, K)
        self.encoder_std = nn.Linear(input_size, K)
        # encoder initialization
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
        mu = self.encoder_mean(x)
        sigma = F.softplus( F.relu(self.encoder_std(x)) - 5 )
        outputs = torch.zeros(self.num_latent, x.shape[0], self.output_size)
        for i in range(self.num_latent):
            latent = Normal(mu, sigma).rsample()
            outputs[i, :, :] = self.decoder(latent)
        outputs = torch.mean(outputs, dim=0)
        return outputs
    
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
    __str__ = adv
    start = time.time()
    
    for i, (images, labels) in enumerate(testloader):
        images = images.to(device)
        images.requires_grad = True
        
        if adv == 'FGM':
            adversaries = FGM(model, images, eps=eps, norm=np.inf)
        elif adv == 'PGD':
            if n_iter > 0:
                adversaries = PGD(model, images, eps=eps, eps_iter=0.01,
                                  nb_iter=n_iter, norm=np.inf)
                __str__ = adv + '_with_{}_iterations_and_{}_eps'.format(n_iter, eps)
            else:
                print("Need proper input for n_iter.")
            
        elif adv == 'CW':
            adversaries = CW(model, images, n_classes=2)
        elif adv == 'SPSA':
            if n_iter > 0:
                adversaries = SPSA(model, images, eps=eps,
                                   nb_iter=n_iter, norm=np.inf,
                                   sanity_checks=False)
                __str__ = adv + '_with_{}_iterations_and_{}_eps'.format(n_iter, eps)
                
            else:
                print("Need proper input for n_iter.")
        else:
            adversaries = images
        
        outputs = model(adversaries).detach().cpu()
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        images = images.detach().cpu()
    
    t = (time.time()-start) // 60
    print()
    print("##################################################################")
    print(__str__)
    print('Computing took ~{:.0f} minutes.'.format(t))
    acc = 100 * correct/total
    print("\nClassified {:.2f} % of test images correctly when perturbed images were created with 12 latent samples.".format(acc))
    print("##################################################################")
    print()
    
    return acc
    
fpath = "/scratch/work/hakkina7/chest_xray/"
print("File path is {}".format(fpath))

# Set the device
dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)
print("Device is {}".format(device))

# Set file name
fname_base = 'resnet50.pth'

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
    
# Inititalize base models
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
print("Pre-trained base models succesfully loaded from {}".format(fpath+fname_base))

betas = [10e-11, 10e-10, 10e-9, 10e-8, 10e-7,
         10e-6, 10e-5, 10e-4, 10e-3, 10e-2]

cws= []


for beta in betas:
    
    print("==================================================================")
    print("BETA = {}".format(beta))
    print("==================================================================")
    
    fname_vib = 'resnet50_vib_{}.pth'.format(beta)
    
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
    cws.append(evaluate(model, testloader, device, fname_vib, adv='CW'))
