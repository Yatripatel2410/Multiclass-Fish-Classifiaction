from keras.preprocessing.image import ImageDataGenerator
from keras.applications.efficientnet import preprocess_input
from utils import create_data_generators

def load_data(data_dir="C:/Users/amulp/OneDrive/Desktop/my_fish_dataset"):
    train_gen, val_gen, test_gen = load_data(data_dir)
    return train_gen, val_gen, test_gen


def load_data(img_size=(224, 224), batch_size=32):
    base_path = 'data'

    datagen_train = ImageDataGenerator(preprocessing_function=preprocess_input)
    datagen_val = ImageDataGenerator(preprocessing_function=preprocess_input)
    datagen_test = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = datagen_train.flow_from_directory(
        f'{base_path}/train',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_gen = datagen_val.flow_from_directory(
        f'{base_path}/val',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_gen = datagen_test.flow_from_directory(
        f'{base_path}/test',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen
def get_class_names(generator):
    return list(generator.class_indices.keys())     
