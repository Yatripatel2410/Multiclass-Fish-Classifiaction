# Multiclass Fish Image Classification

This project classifies fish images into multiple categories using:
- CNN from scratch
- Transfer learning (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0)

## Project Structure
```
Fish_Classification_Project/
│
├── data/                  # Place dataset here (train/ and val/ folders)
├── models/                # Trained models saved here (.h5)
├── train_cnn.py           # Train CNN from scratch
├── train_transfer.py      # Train pre-trained models
├── evaluate_models.py     # Compare models
├── app.py                 # Streamlit web app
├── utils.py               # Helper functions
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
```

## How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Train CNN:
```bash
python train_cnn.py
```
3. Train transfer learning models:
```bash
python train_transfer.py
```
4. Evaluate models:
```bash
python evaluate_models.py
```
5. Run Streamlit app:
```bash
streamlit run app.py
```

## Dataset Format
```
data/train/
    class1/
    class2/
    ...
data/val/
    class1/
    class2/
    ...
```
