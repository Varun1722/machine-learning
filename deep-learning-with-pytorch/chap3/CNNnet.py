import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image,ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES=True

def check_image(object):
    try:
        img = Image.open(object)
        return True
    except:
        return False

# apply transforms
transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

# Load the dataset
train_data_path = "./images/train"
train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=transform,is_valid_file=check_image)

val_data_path = "./images/val"
val_data = torchvision.datasets.ImageFolder(root=val_data_path,transform=transform,is_valid_file=check_image)

test_data_path = "./images/test"
test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=transform,is_valid_file=check_image)

# Define dataloaders
train_data_loader = DataLoader(train_data,64,shuffle=True)
val_data_loader = DataLoader(val_data,64)
test_data_loader = DataLoader(test_data,64)

class CNNnet(nn.Module):
    def __init__(self,num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=11,stride=4,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(64,192,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(192,384,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(384,256,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256*6*6,4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096,num_classes),
        )

    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

cnnNet = CNNnet()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = optim.Adam(cnnNet.parameters(),lr = 0.001)

cnnNet.to(device)

def train(model,optimizer,loss_fn,train_loader,val_loader,epochs,device=device):
    for epoch in range(1,epochs+1):
        training_loss =0.0
        val_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input,target = batch
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss = loss_fn(output,target)
            loss.backward()
            optimizer.step()
            training_loss+=loss.data.item()*input.size(0)
        training_loss/=len(train_loader.dataset)

        model.eval()
        num_correct = 0.0
        num_examples = 0.0
        for batch in val_loader:
            input,target = batch
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            loss=loss_fn(output,target)
            val_loss+=loss.data.item()*input.size(0)
            correct = torch.eq(torch.max(output,dim=1)[1],target)

            num_correct+=torch.sum(correct).item()
            num_examples+=correct.shape[0]
        val_loss/=len(val_loader.dataset)

        print(f"Epoch:{epoch}, training_loss: {training_loss:.2f}, validation_loss: {val_loss:.2f},"
            f"accuracy: {num_correct/num_examples*100}")

train(cnnNet,optimizer,nn.CrossEntropyLoss(),train_data_loader,test_data_loader,50,device)

# Make Predictions
labels = ['fish','cat']
img = Image.open("./images/val/fish/100_1422.JPG")
img = transform(img).to(device)
img = torch.unsqueeze(img,0)

cnnNet.eval()
pred = cnnNet(img)
pred = pred.argmax()




