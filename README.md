# tf_semantic_human_matting

## Usage:

### 1.Clone the repository:

git clone https://github.com/tianmingdu/tf_semantic_human_matting.git

### 2.prepare your own data:

python3/python generate_tfrecord.py

### 3.train your own model:

python3/python train.py

### 4.Evaluation:

python3/python export_inference_graph.py

python3/python predict.py


## Introduction
This repository is a tensorflow version for 'Semantic Human Matting' which has some difference from original paper( https://arxiv.org/pdf/1809.01354.pdf ). The main difference of this code is using simple CNN instead of resnet for the pspnet base due to the limitation of GPU.
