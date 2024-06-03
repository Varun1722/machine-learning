import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image,ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES=True


#First we define a func check_img to check whether PIL can open the image file or not
def check_img(path):
    try:
        im = Image.open(path)
        return True
    except:
        return False

#apply transforms to the loaded datasets
transform = transforms.Compose([transforms.Resize((64,64)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

train_data_path = "./images/train/"
train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=transform,is_valid_file=check_img)

val_data_path = "./images/val/"
val_data = torchvision.datasets.ImageFolder(root=val_data_path,transform=transform,is_valid_file=check_img)

test_data_path = "./images/test/"
test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=transform,is_valid_file=check_img)

#Defining the Dataloaders
train_data_loader = DataLoader(train_data,64,shuffle=True)
val_data_loader = DataLoader(val_data,64)
test_data_loader = DataLoader(test_data,64)


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=12288,out_features=84)
        self.fc2 = nn.Linear(in_features=84,out_features=50)
        self.fc3 = nn.Linear(in_features=50,out_features=2)

    def forward(self,x):
        x= x.view(-1,12288)
        self.x = self.fc1(x)
        self.x = F.relu(self.x)
        self.x = self.fc2(self.x)
        self.x = F.relu(self.x)
        self.x = F.softmax(self.fc3(self.x))
        return self.x

simpleNet = SimpleNet()

optimizer = optim.Adam(simpleNet.parameters(),lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

simpleNet.to(device)
def train(model,optimizer,loss_fn, train_loader,val_loader,epochs,device=device):
    for epoch in range(1,epochs+1):
        training_loss = 0.0
        val_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input, label = batch
            input = input.to(device)
            label = label.to(device)
            output = model(input)
            loss = loss_fn(output,label)
            loss.backward()
            optimizer.step()
            training_loss+= loss.data.item()*input.size(0)
        training_loss/= len(train_loader.dataset)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            input,label = batch
            input = input.to(device)
            label = label.to(device)
            output = model(input)
            loss = loss_fn(output,label)
            val_loss+=loss.data.item()*input.size(0)
            correct = torch.eq(torch.max(output,dim=1)[1],label).view(-1)

            num_correct+=torch.sum(correct).item()
            num_examples+=correct.shape[0]
        val_loss/=len(val_loader.dataset)


    print(f"Epoch :{epoch},training_loss = {training_loss:.2f},validation loss = {val_loss:.2f},"
          f"accuracy = {num_correct/num_examples}")

train(simpleNet,optimizer,nn.CrossEntropyLoss(),train_data_loader,test_data_loader,20,device)

# Making Predictions
labels = ['cat','fish']

img = Image.open("./images/val/fish/100_1422.JPG")
img = transform(img).to(device)
img = torch.unsqueeze(img,0)

simpleNet.eval()
pred = simpleNet(img)
pred = pred.argmax()
print(labels[pred])

# torch.save(simpleNet.state_dict(), "./temp/simplenet")
