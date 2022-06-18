import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torchvision
from torchvision import transforms, models
from torch import nn, optim
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F

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
        sigma = F.softplus(self.encoder_std(x)-5)
        # For loop line 64-66 num_latent times and average over those in eval
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


def train(base_model, vib, epochs, trainloader, device, optimizer):
    """Training loop"""
    scheduler=optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    all_total_losses = []
    all_fit_losses = []
    all_kl_losses = []
    all_epochs = []
    
    start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        vib.train()
        total_losses = []
        fit_losses = []
        kl_losses = []
        
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            outputs = base_model(images)
            outputs, probs, latent, mu, sigma = vib(outputs)
            loss, fit_loss, kl_loss = vib.compute_loss(outputs,
                                                       labels,
                                                       latent,
                                                       mu,
                                                       sigma,
                                                       device
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_losses.append(loss.item())
            fit_losses.append(fit_loss.item())
            kl_losses.append(kl_loss.item())
            
        
        all_total_losses.append(np.mean(total_losses))
        all_fit_losses.append(np.mean(fit_losses))
        all_kl_losses.append(np.mean(kl_losses))
        all_epochs.append(epoch + 1)
        print(
            "\nEpoch {}:\nMean loss {}\nEpoch time ~{} minutes\n".format(
                epoch+1,
                np.mean(total_losses),
                np.round((time.time() - epoch_start) / 60, 0)
            )
        )
        if (i+1) % 2 == 0:
            scheduler.step()

    print(
        "Training finished. Time ~{} hours.".format(
            np.round((time.time()-start) / 3600, 0)
        )
    )
    plt.title('total_loss = fit_loss + beta * kl_loss')
    plt.plot(all_epochs, all_total_losses, 'k', label='total loss')
    plt.plot(all_epochs, all_fit_losses, 'm', label='fit loss')
    plt.plot(all_epochs, all_kl_losses, 'c', label='KL loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(fpath+'loss_resnet_vib_nw.png')
    plt.close()

    return vib


###############################################################################
# MAIN PROGRAM
###############################################################################

fpath = "/scratch/work/hakkina7/chest_xray/"
print("File path is {}".format(fpath))

# Set the device
dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)
print("Device is {}".format(device))

# Set file name
fname_base = 'resnet50_nonweighted.pth'
fname_vib = 'resnet50_vib_nw.pth'
n_epochs = 20

h, w = (224, 224)
transformations = transforms.Compose([
    transforms.Resize((h, w)),
    transforms.ToTensor(),
    transforms.Normalize(
            torch.Tensor([0.0, 0.0, 0.0]), torch.Tensor([1.0, 1.0, 1.0])
    )
])

trainset = torchvision.datasets.ImageFolder(root=fpath+"data/train/",
                                           transform=transformations)

trainloader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=32, shuffle=True
)

# Initialize the model
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

model.fc = Identity()
vib = VIB()
vib.to(device)

# Train model
vib_model = train(
        base_model=model,
        vib=vib,
        epochs=n_epochs,
        trainloader=trainloader,
        device=device,
        optimizer=optim.Adam(list(vib.parameters()),
                             lr=10e-4,
                             betas=(0.5, 0.999)
        )
)

# Save the trained model
torch.save(vib_model.state_dict(), fpath+fname_vib)
print("Model succesfully saved to {}".format(fpath+fname_vib))

