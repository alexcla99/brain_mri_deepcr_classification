from tensorflow import keras
from tensorflow.keras import layers
from params import model_params

def get_model() -> keras.Model :
    """Build a DeepCR-like 3D CNN model."""
    inputs = keras.Input((model_params["input_width"], model_params["input_height"], model_params["input_depth"], 1))
    # First conv block
    x = layers.Conv3D(filters=model_params["filters"][0], kernel_size=model_params["ks"][0], activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=model_params["filters"][1], kernel_size=model_params["ks"][1], activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=model_params["pool_size"])(x)
    # Second conv block
    x = layers.Conv3D(filters=model_params["filters"][2], kernel_size=model_params["ks"][0], activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv3D(filters=model_params["filters"][3], kernel_size=model_params["ks"][1], activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool3D(pool_size=model_params["pool_size"])(x)
    # Output block
    x = layers.AveragePooling3D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(model_params["units"][1], activation="relu")(x)
    x = layers.Dense(model_params["units"][0], activation="relu")(x)
    x = layers.Dropout(model_params["dropout"])(x)
    outputs = keras.activations.sigmoid(x)
    # Build and return the model
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model
