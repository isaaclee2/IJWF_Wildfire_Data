import os
import numpy as np

def data_reader(config):

    #train_frac = config["data"]["train_split"]

    # ===== LOAD DATA =====
    ## this part should be editted in accordance to the format of your data. The example below assumes that the data is stored in a .npz file with keys "X_train", "Y_train", "X_test", "Y_test". You should replace this with the appropriate loading mechanism for your data.
    data_path = config["data"]["data_path"]
    data = np.load(data_path) # or with any command suitable to load your data
    X_train = data["X_train"]
    Y_train = data["Y_train"]
    X_test = data["X_test"]
    Y_test = data["Y_test"]

    # ===== NORMALIZATION (TRAIN ONLY!) =====
    if config["data"]["normalized"]:
        X_train, X_mean, X_scale = normalize_to_unit_range(X_train)
        Y_train, Y_mean, Y_scale = normalize_to_unit_range(Y_train)

        X_test = (X_test - X_mean) / X_scale
        Y_test = (Y_test - Y_mean) / Y_scale

        # ===== SAVE Normalization scale and mean =====
        savedir = config["data"]["save_path_normalization_scale_mean"]
        os.makedirs(savedir, exist_ok=True)
        np.save(os.path.join(savedir, f"X_mean.npy"), X_mean)
        np.save(os.path.join(savedir, f"X_scale.npy"), X_scale)
        np.save(os.path.join(savedir, "Y_mean.npy"), Y_mean)
        np.save(os.path.join(savedir, "Y_scale.npy"), Y_scale)
        return X_train, Y_train, X_test, Y_test ,[X_mean, X_scale, Y_mean, Y_scale]

    return X_train, Y_train, X_test, Y_test , []

def data_reader_eval(config):
    """
    eval_data_path = config["data"]["eval_data_path"]
    eval_data = np.load(eval_data_path) # or with any command suitable to load your data
    Y_eval_samp = eval_data["Y_eval_samp"]
    """
    data_path = config["data"]["data_path"]
    data = np.load(data_path)
    Y_eval_samp = data["Y_test"]  # shape (n_test, 24)
    X_prior = data["X_train"]

    if config["data"]["normalized"]:
        savedir = config["data"]["save_path_normalization_scale_mean"]
        X_mean = np.load(os.path.join(savedir, f"X_mean.npy"))
        X_scale = np.load(os.path.join(savedir, f"X_scale.npy"))
        Y_mean = np.load(os.path.join(savedir, "Y_mean.npy"))
        Y_scale = np.load(os.path.join(savedir, "Y_scale.npy"))

        X_prior = (X_prior - X_mean) / X_scale
        Y_eval_samp = (Y_eval_samp - Y_mean) / Y_scale
        return X_prior, Y_eval_samp, [X_mean, X_scale, Y_mean, Y_scale]
    
    return X_prior, Y_eval_samp, []

def normalize_to_unit_range(Z):
    mean = Z.mean(axis=0)
    Z_centered = Z - mean
    scale = np.max(np.abs(Z_centered), axis=0)

    # avoid division by zero
    scale[scale == 0] = 1.0

    Z_norm = Z_centered / scale
    return Z_norm, mean, scale
