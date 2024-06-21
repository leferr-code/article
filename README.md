# **Disclosing Neonatal Pain in Real-Time: AI-Derived Pain Sign from Continuous Assessment of Facial Expressions**

*This repository contains all code needed to train and run the Deep-Learning models utilized in our research.*

# Requirements

Coded and tested on Python 3.10.

Install necessary python libraries:

```
pip install -r requirements.txt
```

To run most of the codes the iCOPE e UNIFESP datasets are required to be inside a folder Datasets with the original filenames from the dataset authors. To get access to these datasets you must ask permission from its own creators.

```bash
root
├── ...
├── Datasets             # All of the datasets are going to be saved here
│   ├── COPE             # iCOPE images
│   ├── UNIFESP          # UNIFESP images
```

# **Repository Structure**

All code inside folders like `dataloaders`, `models` and `XAI` can be called using Python imports:

```python
from models import VGGNB, NCNN
from XAI import IntegratedGradients
from dataloader import *
```

# **Running the code**

The main codes are on the root directory and can run from the command line or your favorite programming software.

#### **1. [`face_detection.py`](face_detection.py)**

You will need to follow the instructions on [InsightFace](https://github.com/deepinsight/insightface/tree/master/python-package) to download and install the RetinaFace model. It will create a new folder `Datasets\Faces` with the face cropped from each image.

#### **2. [`leave_one_subject_out.py`](leave_some_subject_out.py)**

To run this code you will need the the `Datasets\Faces` folder. It will create 10 folds with Train and Validation sets using the leave-some-subject-out method, each fold will be stored on `Datasets\Folds\{fold_number}`

#### **3. [`data_augmentation.py`](data_augmentation.py)**

To run this code you will need the `Datasets\Folds`. For each training image 20 new images will be created using data augmentation techniques. All images are resized to `512 x 512` and the facial landmarks are augmented as well and saved inside a folder `Datasets\Folds\0\Train\Landmarks` in the [pickle](https://docs.python.org/3/library/pickle.html) format.

#### **4. [`train.py`](train.py)**

After the above steps, you can run the training code. First make sure to create or use one of the existing `.yaml` configuration files inside `models\configs`.

Inside the `.yaml` file you can choose any optimizer from PyTorch and set its hyperparameters on the `optimizer_hyp` field. See `models\configs\config_NCNN.yaml` for a example. Then run:

```text
python train.py --config models\configs\config_NCNN.yaml
```

# Trained models

Trained models can be requested to the corresponding author.