import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image,ImageFile
import torchvision.models as models
from torchvision.models import ResNet50_Weights
ImageFile.LOAD_TRUNCATED_IMAGES=True

def check_image(object):
    try:
        img = Image.open(object)
        return True
    except:
        return False

# apply transforms
transforms = transforms.Compose([transforms.Resize((64,64)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

transfer_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# freezing the training except for batch Normalisation
for name,params in transfer_model.named_parameters():
    if("bn" not in name):
        params.requires_grad = False

# adding the final layer to retrain the pretrained model
transfer_model.fc = nn.Sequential(nn.Linear(transfer_model.fc.in_features,500),
                                  nn.ReLU(),
                                  nn.Dropout(),
                                  nn.Linear(500,2))

# putting model into device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transfer_model.to(device)

# Load the datasets
train = "./images/train"
train_data = torchvision.datasets.ImageFolder(train,transforms,is_valid_file = check_image)
val = "./images/val"
val_data = torchvision.datasets.ImageFolder(val,transforms,is_valid_file = check_image)
test = "./images/test"
test_data = torchvision.datasets.ImageFolder(test,transforms,is_valid_file=check_image)
train_loader = DataLoader(train_data,64,shuffle=True)
val_loader = DataLoader(val_data,64,shuffle = True)
test_loader = DataLoader(test_data,64,shuffle = True)
print(len(val_loader.dataset))

#Hyperparameters
optimizer = optim.Adam(transfer_model.parameters(),lr = 0.001)

# retrain the model on our cat and fish dataset
def train(model,loss_fn,optimizer,train_loader,test_loader,epochs,device):
    for epoch in range(1+epochs):
        training_loss = 0.0
        val_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs,target = batch
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = loss_fn(output,target)
            loss.backward()
            optimizer.step()
            training_loss+=loss.data.item()*inputs.size(0)
        training_loss/=len(train_loader.dataset)

        model.eval()
        num_correct = 0
        num_examples = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs,target = batch
                inputs = inputs.to(device)
                target = target.to(device)
                output = model(inputs)
                loss = loss_fn(output,target)
                val_loss+=loss.data.item()*inputs.size(0)
                correct = torch.eq(torch.max(output,dim=1)[1],target)

                num_correct += torch.sum(correct).item()
                num_examples += correct.shape[0]
            val_loss/=len(test_loader.dataset)
            print(f"Epoch: {epoch}  training loss:{training_loss:.2f}  val loss:{val_loss:.2f}  accuracy:{num_correct/num_examples*100}")

train(transfer_model, nn.CrossEntropyLoss(), optimizer, train_loader, val_loader, 10, device)
