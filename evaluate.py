import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay

# Get file name of weights.
weight_file_name = input("Enter the filename of the model: ")

# Create transformer.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create the model and load the weights.
model = resnet50(weights=None)
model.load_state_dict(torch.load(weight_file_name))

# Define the device based on whether the GPU is available or not. Then, move the model to the device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the loss function and put the model in evaluation mode.
loss_fn = torch.nn.CrossEntropyLoss()
model.eval()

# Load the testing dataset.
valid_ds = datasets.ImageFolder(root="./MRI-Images-of-Brain-Tumor/timri/valid", transform=transform)
valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=True)

# Define the variables to keep track of the total loss and the number of correct predictions.
correct = 0
total = 0
total_loss = 0

# Evaluate the model on the test dataset.
with torch.no_grad():
    for data in valid_loader:
        images, labels = data[0].to(device), data[1].to(device)  # Move the input data to the GPU
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Calculate loss
        loss = loss_fn(outputs, labels)
        total_loss += loss.item()

# Print the accuracy and the average loss.
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
print('Average loss: %.2f' % (total_loss / len(valid_loader)))

# Load the test dataset.
test_ds = datasets.ImageFolder(root="./MRI-Images-of-Brain-Tumor/timri/test", transform=transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=True)

# Define true and prediction label arrays.
true_labels = []
pred_labels = []

# Evaluate the model on the test dataset.
for data in test_loader:
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    _, predicted = torch.max(outputs, 1)
    true_labels.extend(labels.tolist())
    pred_labels.extend(predicted.tolist())

# Calculate F1 Score.
f1 = f1_score(true_labels, pred_labels, average='macro')
print("F1 Score:", f1)

# Calculate confusion matrix.
matrix = confusion_matrix(true_labels, pred_labels)
matrix_display = ConfusionMatrixDisplay(matrix)
matrix_display.plot()
plt.title("Confusion Matrix of the Brain Tumor Detection Model")
plt.savefig("./graphs/confusion_matrix.png")