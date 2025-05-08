
# Real vs. AI-synthesized Images using CNN
This project applies a **Convolutional Neural Network** to predict **Real vs. AI-synthesized Images** using the CIFAKE dataset. The model is trained with a MobileNetV2 base model.
Built and developed by Amith S Patil, Asher Jarvis Pinto, Henry Gladson, Fariza Nuha Farooq and Lavanya Devadiga.

---

## Dataset

- **Source**: (https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
- **Target**: REAL: Images sourced from the CIFAR-10 dataset.
              FAKE: Images generated using Stable Diffusion v1.4.

---

## Workflow

1. **Data Preprocessing**
   - Resized images to `160x160` pixels.
   - Used **ImageDataGenerator** for:
     - **Training data augmentation**: Horizontal flip, zoom (0.2), rotation (15 degrees).
     - **Testing data preprocessing**: Rescaling only.

2. **Modeling**
   - Used **MobileNetV2** (pre-trained on ImageNet) as the base model.
   - Fine-tuned the model by freezing all layers except the last 30.
   - Added a custom classification head:
     - `GlobalAveragePooling2D` → `Dense(128, ReLU)` → `Dropout(0.3)` → `Dense(1, Sigmoid)`.
   - Compiled the model with:
     - Optimizer: **Adam**
     - Loss function: **Binary Cross-Entropy**
     - Metric: **Accuracy**

3. **Training**
   - Trained the model using **train_ds** for training and **test_ds** for validation.
   - Applied **EarlyStopping** if validation loss didn't improve for 3 consecutive epochs.

4. **Evaluation**
   - Metrics:
     - **Accuracy** on test data.
     - **Loss** during training and validation.
   - Visuals:
     - **Training history**: Plots of accuracy and loss.


---

## Evaluation Metrics

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| **FAKE** | 0.91      | 0.96   | 0.93     | 50,000  |
| **REAL** | 0.96      | 0.90   | 0.93     | 50,000  |
| **Accuracy** |       |        | **0.93** | **100,000** |
| **Macro Avg** | 0.93  | 0.93   | 0.93     | 100,000 |
| **Weighted Avg** | 0.93 | 0.93 | 0.93     | 100,000 |

*Table 2: Evaluation metrics of the model performance*


---

## References

Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images.

Bird, J.J. and Lotfi, A., 2024. CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images. IEEE Access.

Real images are from Krizhevsky & Hinton (2009), fake images are from Bird & Lotfi (2024). The Bird & Lotfi study is available here.
