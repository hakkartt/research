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
                 beta=1e-3):
        super(VIB, self).__init__()
        self.input_size = input_size
        self.K = K
        self.output_size = output_size
        self.beta = beta
        self.prior = Normal(
                torch.zeros(1, K), 
                torch.ones(1, K)
        )
        
        # binary classification (cross entropy loss)
        self.fit_loss = nn.NLLLoss(reduction='mean')
        
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
        output = F.softmax(self.decoder(latent), dim=1)
        return output, latent, mu, sigma
    
    def compute_loss(self, pred, true, output_latent, mu, sigma):
        # compute KL loss between encoder output parameters and prior
        Q = Normal(mu.detach(), sigma.detach())
        P = self.prior
        KL_loss = kl_divergence(Q, P).mean()
        fit_loss = self.fit_loss(pred, true)
        # BNN_loss = fit_loss + beta * KL_loss (aka complexity_loss)
        loss = fit_loss + self.beta * KL_loss
        return loss, fit_loss, KL_loss


def train(base_model, vib, epochs, trainloader, device, optimizer):
    """Training loop"""
    scheduler=optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    all_train_losses = []
    all_epochs = []
    
    start = time.time()
    
    for epoch in range(epochs):
        vib.train()
        train_losses = []
        
        for i, (images, labels) in enumerate(trainloader):
            print(i, len(trainloader))
            images, labels = images.to(device), labels.to(device)
            outputs, _ = base_model(images)
            output, latent, mu, sigma = vib(outputs)
            loss, _, _ = vib.compute_loss(output, labels, latent, mu, sigma)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        all_train_losses.append(np.mean(train_losses))
        all_epochs.append(epoch + 1)
        print("\nEpoch {}:\nMean loss {}".format(
                epoch+1,
                np.mean(train_losses)
                )
            )
        if (i+1) % 2 == 0:
            scheduler.step()

    print("Training finished. Time >{} hours.".format((time.time()-start) // 3600))
    plt.title('Validation loss')
    plt.plot(all_epochs, all_train_losses, 'r', label='training loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(fpath+'loss_inception_vib.png')
    plt.close()

    return vib


###############################################################################
# MAIN PROGRAM
###############################################################################

#fpath = "/scratch/work/hakkina7/blindness/"
fpath = "/home/arttu/Documents/research/blindness/"
print("File path is {}".format(fpath))

# Set the device
dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)
print("Device is {}".format(device))

# Set file name
fname_base = 'inceptionV3.pth'
fname_vib = 'inceptionV3_vib.pth'
n_epochs = 1

h, w = (299, 299)
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

print("Training InceptionV3 model for DR blindess classification task.")

# Initialize the model
model = models.inception_v3(pretrained=False)
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
bnn = train(
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
torch.save(bnn.state_dict(), fpath+fname_vib)
print("Model succesfully saved to {}".format(fpath+fname_vib))

