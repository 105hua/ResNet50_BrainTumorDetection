import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
model = resnet50(weights=None)
weights = torch.load(input("Enter the filename of the model: "))
model.load_state_dict(weights)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
model.eval()

test_ds = datasets.ImageFolder(root="./MRI-Images-of-Brain-Tumor/timri/test", transform=transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=True)

correct = 0
total = 0
total_loss = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)  # Move the input data to the GPU
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Calculate loss
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
print('Average loss: %.2f' % (total_loss / len(test_loader)))