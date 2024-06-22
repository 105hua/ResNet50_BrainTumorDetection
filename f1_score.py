import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from sklearn.metrics import f1_score


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

val_ds = datasets.ImageFolder(root="./MRI-Images-of-Brain-Tumor/timri/valid", transform=transform)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=True)

model = resnet50(weights=None)
model.load_state_dict(
    torch.load(input("Enter the filename of the model: "))
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
model.to(device)

model.eval()
true_labels = []
pred_labels = []

for data in val_loader:
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    true_labels.extend(labels.tolist())
    pred_labels.extend(predicted.tolist())

f1 = f1_score(true_labels, pred_labels, average='macro')

print("F1 Score:", f1)