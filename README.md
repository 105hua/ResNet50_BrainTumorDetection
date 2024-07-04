# Brain Tumour Detection with ResNet50

## Introduction

According to Cancer Research UK (2023), a Brain Tumour is a collection of cells in the brain
that have grown out of control. Brain Tumours are usually categorised into two types,
malignant (cancerous) and benign (non-cancerous).

### Benign Tumours

Benign Tumours are non-cancerous, which are tumours that grow more slowly within the Brain.
In addition to this, Benign Tumours are also less likely to come back after treatement, or
spread to other parts of the Brain.

### Malignant Tumours

Malignant Tumours are cancerous, which grow faster in comparison to Benign Tumours.
Malignant Tumours are more likely to come back after their treatment and are more likely
to spread to other parts of the brain.

### Preface

Despite the significant advancements in the research and treatement of Brain Tumours in recent years, unfortunately, the general prognosis for Brain Tumours remains relatively
poor. Cancer Research UK (2023) reports that within a general scope, more than 40% of
people survive malignant tumours for 1 year or more and more than 15% of people survive
for 5 years or more. As a result of this, it is extremely important that the symptoms of
Brain Tumours are recognised and diagnosed as soon as possible.

### Moving into Training

The project aims to achieve this goal through training the ResNet50 architecture on an
Image Dataset of MRI Scans, depicting four different classes:

- **No Tumour:** MRI Scans that fall under this category are normal scans that produce
no concerning results.

- **Glioma:** A malignant type of tumour that begins and grows from the glial 
cells within the brain.

- **Meningioma:** A type of tumour that begins and grows from the meninges
that cover the spinal cord and brain. Most Meningioma's are benign, however,
rarely, they may be cancerous.

- **Pituitary:** A type of tumour that begins and grows from the pituitary
gland at the base of the brain. Similar to Meningiomas, most are benign,
however, they also have the possibility to be cancerous.

The dataset that I have used for this project is the
`MRI-Images-of-Brain-Tumor` dataset, provided by user `PranomVignesh` on
Hugging Face, which you may access for yourself by visiting
[this link](https://huggingface.co/datasets/PranomVignesh/MRI-Images-of-Brain-Tumor).
The dataset may also be cloned by running the `download_dataset.bat` script 
on a Windows System, providing that you have Git installed.

## Training

As mentioned in the Preface within the Introduction section, the ResNet50 model structure
has been used for this project. The model was trained with a batch size of 32 over 40
epochs, using the SGD Optimizer and the CrossEntropyLoss function. The training segment
of the dataset used contains 3,760 images, using an image resolution of 224x224.

## Evaluation

The evaluation of the model took place using the test segment of the dataset, which
contains 538 images. The validation segment of the dataset was not used at any stage, as
none of the steps taken necessitated its use. With these images, the model returned a 97%
accuracy, with an average loss of 0.14, which I believe to be an excellent result. In
addition to this, the F1 Score of the model is 0.978, rounded to 3 decimal places, which is
also a very good result. The final metric that is taken to evaluate the Model is the Mean
Squared Error, which is returned as 0.048, rounded to 3 decimal places.

Following the calculation of the evaluation metrics explained in the previous paragraph,
a confusion matrix is then plotted, which can be seen below with my weight:

<img src="https://i.ibb.co/hBYpCZg/confusion-matrix.png" />

As you can see, the model makes very few incorrect decisions, with the large majority of
predicted labels matching the true label.

## How to Setup

If you'd like to use this project in any means, please follow the guidance below on how
to set the project up.

### Prerequisites

- Python 3.11 or above (This project was developed specifically on version 3.11.9)
- Git

### Steps

- Create a Virtual Environment for the project by running `python -m venv venv`.
- Activate the environment through running `venv\Scripts\activate`. If you are running a
Linux Machine, you may run `source venv/bin/activate`.
- Install PyTorch by following [this link](https://pytorch.org/get-started/locally/).
- Install the rest of the dependencies through running `pip install -r requirements.txt`.
- To download the dataset, please run `py download_dataset.py`. If you encounter any
errors with this script, try running `py alternative_download_dataset.py` to download
the dataset through Git.
- Once the dataset has been obtained, you may train the model by running `py train.py`.
Feel free to experiment with the variables inside of the script.
- If you'd like to evaluate the weights you have trained, you may run
`py evaluate.py`.

This setup guide may be modified in the future to include steps for setting up inferencing
via Gradio.

## References

Cancer Research UK. (2023). Survival for brain and spinal cord tumours. [Online]. Cancer Research UK. Last Updated: 16 June 2023. Available at: https://www.cancerresearchuk.org/about-cancer/brain-tumours/survival [Accessed 18 June 2024].

Cancer Research UK. (2023). What are Brain Tumours?. [Online]. Cancer Research UK. Last Updated: 18 January 2023. Available at: https://www.cancerresearchuk.org/about-cancer/brain-tumours/about [Accessed 18 June 2024].