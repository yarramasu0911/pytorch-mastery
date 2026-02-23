# PyTorch Mastery: 10-Project Roadmap

A hands-on, progressive journey through PyTorch — from manual gradient descent to production-level deep learning. Each project builds on the previous one, covering all major concepts.

## Projects

| # | Project | Key Concepts | Dataset | Status |
|---|---------|-------------|---------|--------|
| 1 | [Linear Regression](./pytorch_learning/LinearRegression/01_Linear_Regression.ipynb) | Tensors, autograd, `nn.Module`, MSELoss, SGD vs Adam, train/test split, model saving | Synthetic | ✅ |
| 2 | [Binary Classification](./pytorch_learning/BinaryClassification/02_Binary_Classification.ipynb) | `BCELoss`, sigmoid, accuracy, precision/recall, confusion matrix, feature scaling | Breast Cancer | ✅ |
| 3 | [Multi-class Classification (Deep MLP)](./pytorch_learning/MultiClassification/03_Multiclass_Classification.ipynb) | `CrossEntropyLoss`, dropout, batch norm, LR schedulers, early stopping | Fashion-MNIST | ✅ |
| 4 | [CNN Image Classification](./pytorch_learning/CNNClassification/04_CNN_Image_Classification.ipynb) | `Conv2d`, `MaxPool2d`, data augmentation, `DataLoader`, GPU training | CIFAR-10 | ✅ |
| 5 | [Transfer Learning](./pytorch_learning/TransferLearning/05_Transfer_Learning.ipynb) | Pretrained ResNet18, freezing layers, fine-tuning, different LRs | CIFAR-10 | ✅ |
| 6 | [Text Classification (RNN/LSTM)](./pytorch_learning/TextClassification/06_Text_Classification_LSTM.ipynb) | `Embedding`, `LSTM`, bidirectional, sequence padding, gradient clipping | IMDB Sentiment | ✅ |
| 7 | Transformer from Scratch | Self-attention, multi-head attention, positional encoding, masking | Translation | ⬜ |
| 8 | Object Detection | Faster R-CNN, bounding boxes, IoU, mAP | Pascal VOC | ⬜ |
| 9 | Autoencoder & VAE | Encoder-decoder, latent space, KL divergence, generative models | MNIST/CelebA | ⬜ |
| 10 | End-to-End Pipeline | PyTorch Lightning, TensorBoard, hyperparameter tuning, ONNX export | Any | ⬜ |

## Setup

### Local (VS Code + CPU)
```bash
python -m venv pytorch_env

# Windows:
pytorch_env\Scripts\activate
# Mac/Linux:
source pytorch_env/bin/activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install jupyter matplotlib numpy pandas scikit-learn
```

### Google Colab (GPU — for Projects 4+)
```python
# PyTorch and GPU come pre-installed on Colab
# Additional dependencies:
!pip install datasets --quiet          # For Project 6 (IMDB dataset)
!pip install scikit-learn --quiet      # For evaluation metrics
```
**Note:** Set Runtime → Change runtime type → T4 GPU

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

**Prasanth** — MS in Artificial Intelligence, Northeastern University (Roux Institute)
