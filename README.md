1.Project Introduction
SIFCNet is a low-light image enhancement network designed to enhance low-light images by semantic–illumination fusion and color restoration.
2.Environment Setup
2.1Create a Virtual Environment
Create and activate a virtual environment:
conda create -n SIFCNet python=3.8
conda activate SIFCNet
2.2Install Dependencies
Install all necessary dependencies:
pip install -r requirements.txt
2.3Prepare Datasets
Download and prepare the training and validation datasets:
Training Dataset: Download the training dataset and place it in basicsr/data/LOLv1/Train/
Validation Dataset: Download the validation dataset and place it in basicsr/data/LOLv1/Test/
Download Links:
Training Dataset:https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?pli=1
Validation Dataset:https://drive.google.com/drive/folders/1rWa_WRX5bqlW2HnBNMUGFKWrou7gIQpO
3.Training and Testing
3.1Training
To train the model, use the following command:
python  basicsr/train.py --opt Options/SIFCNet_LOL_v1.yml
3.2Testing
To test the model, use the following command:
python  basicsr/test.py --opt Options/SIFCNet_LOL_v1.yml
