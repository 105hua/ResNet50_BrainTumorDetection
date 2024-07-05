import torch
import os
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error
from safetensors.torch import load_model

# Get file name of weights.
weight_file_name = input("Enter the filename of the safetensors file: ")

# Create transformer.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Create the model and load the weights.
model = resnet50(weights=None)
load_model(model, weight_file_name)

# Define the device based on whether the GPU is available or not. Then, move the model to the device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the loss function and put the model in evaluation mode.
loss_fn = torch.nn.CrossEntropyLoss()
model.eval()

# Load the testing dataset.
test_ds = datasets.ImageFolder(root="./timri/test", transform=transform)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=True)

# Define the variables to keep track of the total loss and the number of correct predictions.
correct = 0
total = 0
total_loss = 0

# Evaluate the model on the test dataset.
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

# Print the accuracy and the average loss.
print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
print('Average loss: %.2f' % (total_loss / len(test_loader)))

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

# Mean Squared Error
mse = mean_squared_error(true_labels, pred_labels)
print("Mean Squared Error:", mse)

# Calculate confusion matrix.
matrix = confusion_matrix(true_labels, pred_labels)
matrix_display = ConfusionMatrixDisplay(matrix)
matrix_display.plot()
plt.title("Confusion Matrix of the Brain Tumor Detection Model")
os.makedirs("./graphs", exist_ok=True) # Create the directory if it doesn't exist.
plt.savefig("./graphs/confusion_matrix.png")