import torch
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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

matrix = confusion_matrix(true_labels, pred_labels)
matrix_display = ConfusionMatrixDisplay(matrix)
matrix_display.plot()
plt.title(f"Confusion Matrix of the Brain Tumor Detection Model")
plt.savefig("./graphs/confusion_matrix.png")