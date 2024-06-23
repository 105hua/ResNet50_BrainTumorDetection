import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_ds = datasets.ImageFolder(root="./timri/train", transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

model = resnet50(weights=None)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.to(device)

num_of_epochs = 40

for epoch in range(num_of_epochs):
    running_loss = 0.0
    for i, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_of_epochs}", unit="batch")):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Loss: {running_loss / len(train_loader)}")

print("Finished Training!")

print("Saving model...")
torch.save(model.state_dict(), f"sgd_{num_of_epochs}e.pth")
print("Model saved as model.pth")