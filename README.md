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

## Results

The configuration that has been used to train this model is as follows:

- **Batch Size:** 32
- **Weights:** None
- **Loss Function:** Cross Entropy
- **Optimizer:** SGD *(Learning Rate: 0.001 & Momentum: 0.9)*
- **Epochs:** 40

With the use of this configuration, I was able to achieve some excellent results in the
evaluation process:

- **Accuracy:** 97%
- **Average Loss:** 0.13
- **F1 Score:** 0.977

I plan to add more in-depth evaluations to my scripts at some stages in the future.

## References

Cancer Research UK. (2023). Survival for brain and spinal cord tumours. [Online]. Cancer Research UK. Last Updated: 16 June 2023. Available at: https://www.cancerresearchuk.org/about-cancer/brain-tumours/survival [Accessed 18 June 2024].

Cancer Research UK. (2023). What are Brain Tumours?. [Online]. Cancer Research UK. Last Updated: 18 January 2023. Available at: https://www.cancerresearchuk.org/about-cancer/brain-tumours/about [Accessed 18 June 2024].