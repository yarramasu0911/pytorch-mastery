# PyTorch Mastery: 10-Project Roadmap

A hands-on, progressive journey through PyTorch — from manual gradient descent to production-level deep learning. Each project builds on the previous one, covering all major concepts needed for ML engineering interviews.

## Projects

| #   | Project                                                                                                             | Key Concepts                                                                         | Dataset        | Status |
| --- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | -------------- | ------ |
| 1   | [Linear Regression](./pytorch_learning/LinearRegression/01_Linear_Regression.ipynb)                                 | Tensors, autograd, `nn.Module`, MSELoss, SGD vs Adam, train/test split, model saving | Synthetic      | ✅     |
| 2   | [Binary Classification](./pytorch_learning/BinaryClassification/02_Binary_Classification.ipynb)                     | `BCELoss`, sigmoid, accuracy, precision/recall, confusion matrix, feature scaling    | Breast Cancer  | ✅     |
| 3   | [Multi-class Classification (Deep MLP)](./pytorch_learning/MultiClassification/03_Multi_Class_Classification.ipynb) | `CrossEntropyLoss`, dropout, batch norm, LR schedulers, early stopping               | Fashion-MNIST  | ✅     |
| 4   | [CNN Image Classification](./pytorch_learning/CNNClassification/04_CNN_Image_Classification.ipynb)                  | `Conv2d`, `MaxPool2d`, data augmentation, `DataLoader`, GPU training                 | CIFAR-10       | ✅     |
| 5   | Transfer Learning                                                                                                   | Pretrained ResNet/EfficientNet, freezing layers, fine-tuning                         | Custom         | ⬜     |
| 6   | Text Classification (RNN/LSTM)                                                                                      | `Embedding`, `LSTM`, `GRU`, sequence padding, variable-length inputs                 | IMDB Sentiment | ⬜     |
| 7   | Transformer from Scratch                                                                                            | Self-attention, multi-head attention, positional encoding, masking                   | Translation    | ⬜     |
| 8   | Object Detection                                                                                                    | Faster R-CNN, bounding boxes, IoU, mAP                                               | Pascal VOC     | ⬜     |
| 9   | Autoencoder & VAE                                                                                                   | Encoder-decoder, latent space, KL divergence, generative models                      | MNIST/CelebA   | ⬜     |
| 10  | End-to-End Pipeline                                                                                                 | PyTorch Lightning, TensorBoard, hyperparameter tuning, ONNX export                   | Any            | ⬜     |

## Setup

```bash
# Create virtual environment
python -m venv pytorch_env

# Activate
# Windows:
pytorch_env\Scripts\activate
# Mac/Linux:
source pytorch_env/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install jupyter matplotlib numpy pandas scikit-learn
```

## Tech Stack

- **Framework:** PyTorch 2.10+
- **Environment:** VS Code + Jupyter
- **Visualization:** Matplotlib
- **Python:** 3.10+

## What This Covers

- Tensors, autograd, and computation graphs
- Neural network architectures (MLP, CNN, RNN, LSTM, Transformer)
- Transfer learning and fine-tuning
- Regularization (dropout, batch norm, early stopping)
- Optimizers (SGD, Adam) and learning rate scheduling
- Loss functions (MSE, BCE, CrossEntropy)
- Data pipelines (Dataset, DataLoader, transforms)
- Model evaluation, saving, and deployment
- Generative models (VAE)
- Production tooling (PyTorch Lightning, TensorBoard, ONNX)

## Author

**Prasanth Yarramasu**
