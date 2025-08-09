import os
from data_loader import load_data
from cnn_model import create_cnn
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd

train_gen, val_gen, _ = load_data()
model = create_cnn((224, 224, 3), train_gen.num_classes)

os.makedirs("models", exist_ok=True)
checkpoint = ModelCheckpoint("models/cnn_model.h5", monitor="val_accuracy", save_best_only=True)
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

history = model.fit(train_gen, validation_data=val_gen, epochs=20, callbacks=[checkpoint, early_stop])

pd.DataFrame(history.history).to_csv("models/training_history.csv", index=False)
print("✅ CNN training completed and saved.")

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save model after training
model.save(os.path.join("models", "cnn_model.h5"))
print("✅ Model saved to models/cnn_model.h5")
