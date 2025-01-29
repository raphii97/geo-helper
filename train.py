import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

#Dataset preparation
class GeoDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = f"{self.img_dir}/{index}.png"
       
        image = Image.open(img_path)#.convert("RGB")
        y1, y2 = self.data.iloc[index, :]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor([y1, y2], dtype=torch.float32)



transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

train_dataset = GeoDataset("coords.csv", "dataset", transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#print(train_dataset[0])




#CNN Model
class GeoCNN(nn.Module):
    def __init__(self):
        super(GeoCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1) # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x) # no activation for continuous values
        return x
    

#training + performance evaluation + save model
model = GeoCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, targets in train_loader:
        images, targets = images.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    losses.append(running_loss / len(train_loader))
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("Training complete!")
torch.save(model.state_dict(), "geo_cnn.pth")
print("Model saved.")

plt.plot(range(1, epochs + 1), losses, marker='o', linestyle='-', color='b')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss over Epochs")
plt.grid()
plt.show()