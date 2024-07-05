import torch
import os
import gradio as gr

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50

model = resnet50(weights=None)

weight_file_name = input("Enter the filename of the safetensors file: ")

model.load_state_dict(
    torch.load(
        os.path.join(os.getcwd(), weight_file_name)
    )
)

model.eval()

print("Defining device...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("Using device:", device)

classes = {
    0: "glioma",
    1: "meningioma",
    2: "no-tumor",
    3: "pituitary"
}

def inference(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)
    output = model(image)
    _, predicted = torch.max(output, 1)
    predicted_item = int(predicted.item())
    return classes[predicted_item]


iface = gr.Interface(
    fn=inference,
    inputs="file",
    outputs="text",
    title="Brain Tumor Classifier",
    description="This is a simple image classifier that can classify brain tumor images into 4 categories."
)

iface.launch()