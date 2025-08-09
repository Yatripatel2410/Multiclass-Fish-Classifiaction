# train_model.py
import os
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from cnn_model import create_cnn
from data_loader import load_data

# === Load Data ===
train_gen, val_gen, _ = load_data()
num_classes = train_gen.num_classes

# === Compute Class Weights for Imbalanced Data ===
labels = train_gen.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights_dict = dict(enumerate(class_weights))

# === Build CNN Model ===
model = create_cnn((224, 224, 3), num_classes)

# === Create models directory if not exists ===
os.makedirs("models", exist_ok=True)

# === Callbacks ===
checkpoint = ModelCheckpoint(
    'models/cnn_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1
)

# === Train Model ===
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    class_weight=class_weights_dict,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# === Save Training History ===
history_df = pd.DataFrame(history.history)
history_df.to_csv("models/training_history.csv", index=False)
print("✅ Training completed successfully. Model and history saved in 'models/'")