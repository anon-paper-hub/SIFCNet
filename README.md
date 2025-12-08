# Project Introduction
**SIFCNet** is a low-light image enhancement network designed to enhance low-light images by semantic–illumination fusion and color restoration.

# Environment Setup
## Create a Virtual Environment
Create and activate a virtual environment:
```python
conda create -n SIFCNet python=3.8
conda activate SIFCNet
```
## Install Dependencies
Install all necessary dependencies:
```python
pip install -r requirements.txt
```
## Prepare Datasets
Download and prepare the training and validation datasets:
Training Dataset: Download the training dataset and place it in basicsr/data/LOLv1/Train/
Validation Dataset: Download the validation dataset and place it in basicsr/data/LOLv1/Test/

Download Links:
Training Dataset:[Google Drive](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?pli=1) 
Validation Dataset:[Google Drive](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?pli=1)

# Training and Testing
## Training
To train the model, use the following command:
```
python  basicsr/train.py --opt Options/SIFCNet_LOL_v1.yml
```
## Testing
To test the model, use the following command:
```
python  basicsr/test.py --opt Options/SIFCNet_LOL_v1.yml
```
