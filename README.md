# AI Face Mask Detector
In this project we implemented CNN in 2 different way (Using PyTorch and Tensorflow)

## Folder and File Instructions and descriptions.
2 code files with the extension of .py . One of the file (cnn_pytorch.py) represents CNN model using PyTorch, As well as this file contains data processing and data loading to the variable. Also has the code for the evaluation and confusion matrix. The other file (cnn_tensor.py) represents CNN model using Tensorflow, As well as also , this file contains data processing and data loading to the variable. Also has the code for the evaluation and confusion matrix.

2 Output files contains the dump of the 2 models output (Full result).

1 Readme.md file which represents the description and Required libris as well as the instruction to run the code.

1 project report file. It contains the overall description of the project

1 "import_data" folder contains all the data sets.

1 torch.pt is the save model for pytorch. if you have this in your folder it will avoid training and will load the trained model.


Note: If you are cloning the repository you don't need to worry about the data. But you are downloading from different place other than the github. Please download the whole data set and replace the "import_data" folder with the downloaded "import_data" folder . Data link: https://hasibulhuq.com/ai/  or https://www.kaggle.com/mdhasibulhuq/ai-face-with-mask-without-mask-and-non-huma. This images are for research and academic purpose. For the copywrite issue Please see the project report file. The original sources are provided there.


## Required modules

Make sure you have this module installed in your pc for running the cnn_pytorch.py (Please see the above description about the file )
```
Python 3.7.3
from torch.utils.data import Dataset, DataLoader
torch
torchvision
multiprocessing
time
pandas
os
cv2
random
sklearn.model_selection import train_test_split
numpy as np
sklearn.preprocessing import LabelEncoder
sklearn.metrics import classification_report
sklearn.metrics import confusion_matrix
```

Make sure you have this module installed in your pc for running the cnn_tensor.py (Please see the above description about the file )

```
Python 3.7.3
from torch.utils.data import DataLoader
torch
torchvision
sklearn.preprocessing
sklearn.model_selection
pandas
multiprocessing
time
os
cv2
csv
random
tensorflow as tf
numpy as np
tensorflow.keras import datasets, layers, models , utils
sklearn.metrics import classification_report
sklearn.metrics import confusion_matrix
```


## How to run code

To Run CNN using PyTorch

```
python .\cnn_pytorch.py
```

To Run CNN using Tensorflow

```
python .\cnn_tensor.py
```

## Note
Code may generate warnings. We handeled some errors using try catch and printed the error. If you see error message which does not stop your program. don't bother. It will finally give you the answer .  

## Reference
Please check the project report
