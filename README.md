# CheXpert Challenge

### Data
kaggle: https://www.kaggle.com/datasets/ashery/chexpert

(Note: This is a subet of the original cheXpert dataset from standford. The full dataset can be found at: https://aimi.stanford.edu/datasets/chexpert-chest-x-rays)
```
CHEXPERT_CLASSES = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]
```

### Prerequisites
```
pip install -r requirements.txt
```

### Project

#### Data Agumentation
1. [CVAE](https://arxiv.org/abs/1908.09008#:~:text=Prediction%20of%20future%20states%20of,results%20across%20different%20evaluation%20metrics.)
    - Code: https://medium.com/data-science/conditional-variational-autoencoders-with-learnable-conditional-embeddings-e22ee5359a2a
    - Purpose:
        - The Complexity: Implement a VQ-VAE (Vector Quantized-VAE) or a Conditional VAE (C-VAE). This allows you to generate images specific to a class (e.g., "Generate a lung with Pneumonia") rather than random samples.
        - The Validation: Your VAE specialist shouldn't just generate images; they must prove they are statistically similar to the real ones using FID (Fréchet Inception Distance) or t-SNE plots.
        - Metric Goal: Show that adding VAE data actually improves the F1-Score of the minority classes compared to simple oversampling.

#### Models
1. [ResNet50](https://arxiv.org/abs/1512.03385)
2. [ResNet-SE](https://arxiv.org/pdf/1709.01507)
    - Code: https://apxml.com/courses/cnns-for-computer-vision/chapter-5-attention-transformers-vision/implementing-attention-blocks-practice
3. [ResNet-CBAM](https://arxiv.org/abs/1807.06521)
4. Ensemble Model (1 + 2 + 3):
    - Transfer learning
    - Pass validation images through all three to get their predicted probabilities ($3 \text{ models} \times 3 \text{ classes} = 9 \text{ features}$)
    - Train a gating network (a simple MLP) to learns which classifier to trust for specific types of images and utilize the idea of Monte Carlo Method to evaluate the uncertian of a prediction

#### Monte Carlo Method
Standard models give a single probability (e.g., "80% Pneumonia"). But is that 80% because the model is confident, or is it a random guess due to noisy data?
- Idea: Calculate the variance (standard deviation) between the probabilities outputted by the three different models. This achieves the exact same goal as MC Dropout (flagging images the AI is unsure about) without needing to alter the pre-trained architectures.
- The Output: If the 20 results are all similar, the model is "Certain." If they vary wildly, the model is "Uncertain."
- The Clinical Value: In a real hospital, an "Uncertain" flag would trigger an automatic referral to a human radiologist.

##### Grad-CAM (XAI)
- Source: https://github.com/jacobgil/pytorch-grad-cam
- Focus on interpretability of the computer vision and highlight the area where the model is looking for on an image

#### User Interface
- Source: https://streamlit.io/
- A Predicted Label
- Present how the model is looking into the image
- The Confidence Score (Enemsble Model)
- The Uncertainity Warning (e.g., When high uncertainty detected, specialists are needed)

### Training & Testing
- Training:
```
# Training a model at first time or continue to train a model
python .\trainModels.py -c .\model_confs\ResNetCBAM.yaml -p U-Ones
```

- Testing:
```
# Automatically load your previsouly trained model
python .\testModels.py -c .\model_confs\ResNetCBAM.yaml -p U-Ones
```

You can replace -c (.\model_confs\Your Model) and -p (Your Policy)

### Pretrained Models

- [Models Without CVAE Data](https://drive.google.com/drive/folders/1PqiQ_yJkTNa8mqyL2yFbUux5TuqPifht?usp=sharing)

- [Models With CVAE Data](https://drive.google.com/drive/folders/1Fer3AeZOcyfBldkGkmzNcMQp1akBqKlm?usp=drive_link)

### Summary




