# Model parameters
model_params = {
    "input_width": 113,
    "input_height": 137,
    "input_depth": 113,
    "filters": [32, 64, 128, 256],
    "ks": [3, 5],
    "pool_size": 2,
    "units": 1,
    "dropout": .2
}

# Experiment parameters
run_params = {
    "batch_size": 16,
    "loss": "binary_crossentropy",
    "lr": .016,
    "patience": 25, #100
    "epochs": 100
}