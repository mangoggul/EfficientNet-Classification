# EfficientNet PyTorch

## Source : [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
### Quickstart

Install with `pip install efficientnet_pytorch` and load a pretrained EfficientNet with:
```python
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')
```



### Overview
This repository is easily written how to use [EfficientNet](https://arxiv.org/abs/1905.11946)


You can easily:
 * Load pretrained EfficientNet models
 * Use EfficientNet models to Train your own Custom DataSet
 * Evaluate EfficientNet models by your own Images










### About EfficientNet PyTorch

EfficientNet PyTorch is a PyTorch re-implementation of EfficientNet. It is consistent with the [original TensorFlow implementation](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet), such that it is easy to load weights from a TensorFlow checkpoint. At the same time, we aim to make our PyTorch implementation as simple, flexible, and extensible as possible.

If you have any requests or questions, leave them as GitHub issues.

### Installation

Install via pip:
```bash
pip install efficientnet_pytorch
```

Or install from source:
```bash
git clone https://github.com/mangoggul/efficientNet-Classification.git
cd EfficientNet-Pytorch
```

### Usage

#### Loading pretrained models

Load an EfficientNet:
```python
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_name('efficientnet-b0')
```

Load a pretrained EfficientNet:
```python
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')
```

Details about the models are below:

|    *Name*         |*# Params*|*Top-1 Acc.*|*Pretrained?*|
|:-----------------:|:--------:|:----------:|:-----------:|
| `efficientnet-b0` |   5.3M   |    76.3    |      ✓      |
| `efficientnet-b1` |   7.8M   |    78.8    |      ✓      |
| `efficientnet-b2` |   9.2M   |    79.8    |      ✓      |
| `efficientnet-b3` |    12M   |    81.1    |      ✓      |
| `efficientnet-b4` |    19M   |    82.6    |      ✓      |
| `efficientnet-b5` |    30M   |    83.3    |      ✓      |
| `efficientnet-b6` |    43M   |    84.0    |      ✓      |
| `efficientnet-b7` |    66M   |    84.4    |      ✓      |


#### Example: Classification

Below is a simple, complete example. It may also be found as a jupyter notebook in `examples/simple` or as a [Colab Notebook](https://colab.research.google.com/drive/1Jw28xZ1NJq4Cja4jLe6tJ6_F5lCzElb4).

We assume that in your current directory, there is a `img.jpg` file and a `labels_map.txt` file (ImageNet class names). These are both included in `examples/simple`.

```python
import json
from PIL import Image
import torch
from torchvision import transforms

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')

# Preprocess image
tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
img = tfms(Image.open('img.jpg')).unsqueeze(0)
print(img.shape) # torch.Size([1, 3, 224, 224])

# Load ImageNet class names
labels_map = json.load(open('labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

# Classify
model.eval()
with torch.no_grad():
    outputs = model(img)

# Print predictions
print('-----')
for idx in torch.topk(outputs, k=5).indices.squeeze(0).tolist():
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))
```

#### Example: Feature Extraction

You can easily extract features with `model.extract_features`:
```python
from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')

# ... image preprocessing as in the classification example ...
print(img.shape) # torch.Size([1, 3, 224, 224])

features = model.extract_features(img)
print(features.shape) # torch.Size([1, 1280, 7, 7])
```

#### Example: Export to ONNX

Exporting to ONNX for deploying to production is now simple:
```python
import torch
from efficientnet_pytorch import EfficientNet

model = EfficientNet.from_pretrained('efficientnet-b1')
dummy_input = torch.randn(10, 3, 240, 240)

model.set_swish(memory_efficient=False)
torch.onnx.export(model, dummy_input, "test-b1.onnx", verbose=True)
```

[Here](https://colab.research.google.com/drive/1rOAEXeXHaA8uo3aG2YcFDHItlRJMV0VP) is a Colab example.


#### ImageNet

See `examples/imagenet` for details about evaluating on ImageNet.

## Own Custom data Training
**Just note** : my data was [Pc parts](https://www.kaggle.com/datasets/asaniczka/pc-parts-images-dataset-classification)
### First
Prepare your Dataset. The Dataset Structure needs to be 

### Directory Structure Explanation:

- **data/**: Root directory containing all data related to training and validation.
  
  - **train/**: Training dataset directory.
    - **class1/**: Directory containing images belonging to class 1.
      - `image1.jpg`, `image2.jpg`, ...: Actual image files of class 1.
    - **class2/**: Directory containing images belonging to class 2.
      - `image1.jpg`, `image2.jpg`, ...: Actual image files of class 2.
    - Additional directories for other classes if your dataset has more classes.

  - **val/**: Validation dataset directory (similar structure as `train`).
    - **class1/**: Directory containing images for validation from class 1.
      - `image1.jpg`, `image2.jpg`, ...: Actual image files for validation of class 1.
    - **class2/**: Directory containing images for validation from class 2.
      - `image1.jpg`, `image2.jpg`, ...: Actual image files for validation of class 2.
    - Additional directories for other classes if needed.

### Usage:

Ensure your dataset follows this structure to properly load and train with EfficientNet or any other deep learning model. Adjust the number of classes and the number of images per class based on your specific dataset requirements.

---

### Second
make your own labels.map.txt file . 
The file Architecture must be like this. 

{"0": "cpu", "1": "gpu", "2": "hdd"} 

If you want to classify 3 classes then like this. So if you want more classes to classify. 
It will be like

{"0": "cpu", "1": "gpu", "2": "hdd", "3" : "mango", "4" : "ggul", ................., "N-1" : "className" } 

---


### Third
Now you can Train your dataset. Run main.py file to Train your Data. Consider some arguments in the file. 
The simple way to just train is
```python
python main.py /path/to/dataset --arch efficientnet-b0 --pretrained 
```
---


### Fourth
Whenever your training is Done. Go to inference.ipynb and modify some codes. 

```
# Create the model (use the same architecture as the trained model)
model = EfficientNet.from_name('efficientnet-b0')  # or the appropriate version

# Load your trained weights
checkpoint = torch.load('../examples/imagenet/model_best.pth.tar') #여기 customData로 train 한 가중치 넣으면 됩니다. 
```
you need to modify checkpoint to your own pth file. 
And last change image to your own image. 

Then It will work properly. Happy Classification !!

