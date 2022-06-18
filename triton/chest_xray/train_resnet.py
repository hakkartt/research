import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import torchvision
from torchvision import transforms, models
from torch import nn, optim
import torch.nn.functional as F
import sklearn.metrics as metrics

fpath = "/scratch/work/hakkina7/chest_xray/"
print("File path is {}".format(fpath))

# Set the device
dev = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(dev)
print("Device is {}".format(device))

# Set file name
fname = 'resnet50.pth'
n_epochs = 200

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

def train(model, epochs, trainloader, valloader, device, optimizer, loss_fn):
    """Training loop"""
    
    best_model = None
    min_loss = 1000
    all_val_losses = []
    all_train_losses = []
    all_epochs = []
    
    start = time.time()
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            _, pred = torch.max(outputs, 1)
            correct += (labels == pred).sum().item()
            total += labels.size(0)
            loss = loss_fn(outputs, labels)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()
        
        all_train_losses.append(np.mean(train_losses))
        acc = 100 * correct/total
        print("\nEpoch {}:\nMean loss {} - Correctly labeled training images {:.2f}%".format(epoch+1,
                                                                                           np.mean(train_losses),
                                                                                           acc))
        
        model.eval()
        val_losses = []
        correct = 0
        total = 0
        true = torch.zeros(0, dtype=torch.long, device=device)
        preds = torch.zeros(0, dtype=torch.long, device=device)
        probs = torch.zeros(0, dtype=torch.float, device=device)
        
        with torch.no_grad():
        
            for i, (images, labels) in enumerate(valloader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, pred = torch.max(outputs, 1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
                loss = loss_fn(outputs, labels)
                val_losses.append(loss.item())
                true = torch.cat([true, labels.view(-1)])
                preds = torch.cat([preds, pred.view(-1)])
                prob = (F.softmax(outputs, dim=1))[:, 1]
                probs = torch.cat([probs, prob.view(-1)])
                
        true, preds, probs = true.detach().cpu(), preds.detach().cpu(), probs.detach().cpu()
                
        acc = 100 * correct/total
        print("Classified {:.2f} % of validation images correctly.".format(
            acc
        ))
        print("Precision on validation set: {:.2f}".format(
            metrics.precision_score(true.numpy(), preds.numpy())
        ))
        print("Recall on validation set: {:.2f}".format(
            metrics.recall_score(true.numpy(), preds.numpy())
        ))
        print("F1-score on validation set: {:.2f}".format(
            metrics.f1_score(true.numpy(), preds.numpy())
        ))
        fpr, tpr, _ = metrics.roc_curve(true.numpy(), probs.numpy(), pos_label=1)
        print("AUC score on validation set: {:.2f}".format(
            metrics.auc(fpr, tpr)
        ))
        print("Confusion matrix on validation set:")
        print(pd.DataFrame(metrics.confusion_matrix(true.numpy(), preds.numpy()), 
                           columns=['pred as NEG', 'pred as POS'], 
                           index=['true class NEG', 'true class POS']))
        print()
        
        # Save best model to be returned
        current_loss = np.mean(val_losses)
        print("Current validation loss: {:.3f}".format(current_loss))
        all_val_losses.append(current_loss)
        all_epochs.append(epoch + 1)
        if current_loss < min_loss:
            best_model = model
            min_loss = current_loss

    print("Training finished. Time >{} hours.".format((time.time()-start) // 3600))
    plt.title('Validation loss')
    plt.plot(all_epochs, all_val_losses, 'b', label='validation loss')
    plt.plot(all_epochs, all_train_losses, 'r', label='training loss')
    plt.legend(loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig(fpath+'loss_resnet.png')
    plt.show()

    return best_model


print("Training ResNet50 model for Chest X-ray Pneumonia classification task.")

# Initialize the model
model = models.resnet50(pretrained=False)
n_labels = 2
n_inputs = model.fc.in_features
model.fc = nn.Linear(n_inputs, n_labels)
model = model.to(device)

# Calculate weights for the weighted loss function. We need the weighted loss,
# because of the class imbalance that is present on the data sets. If
# weighted loss fails, try focal loss with alpha=0.25 and gamma=2.
n_samples = [1266, 3418]
weights = [1 / (x / sum(n_samples)) for x in n_samples]  # More weight on the samples that appear less
weights = torch.FloatTensor(weights).to(device)
print("Since there is class imbalance present in some of the data sets, use the following weights in the loss function:")
print(weights)
print()

# Train model
model = train(model=model,
              epochs=n_epochs,
              trainloader=trainloader,
              valloader=valloader,
              device=device,
              optimizer=optim.SGD(list(model.parameters()),
                                  lr=1e-4,
                                  momentum=0.9),
              loss_fn=nn.CrossEntropyLoss(weight=weights))

# Save the trained model
torch.save(model.state_dict(), fpath+fname)
print("Model succesfully saved to {}".format(fpath+fname))

