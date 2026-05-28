# trainer.py
import os
import sys
import time
import logging
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from optimal_transport import OTPlanSampler
# plotting style is optional; uncomment if you have scienceplots
# import scienceplots
# plt.style.use(['science', 'no-latex'])

# reuse your Flow from flow.py
from flow import Flow, Flow_bridge
from tqdm import tqdm
from data_reader import data_reader, data_reader_eval, normalize_to_unit_range

# -------------------------
# Utilities
# -------------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def create_logger(log_file_path):
    logger = logging.getLogger("exp_logger")
    logger.setLevel(logging.INFO)
    # avoid duplicate handlers if called multiple times
    if logger.handlers:
        logger.handlers = []

    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# -------------------------
# Train both flows (eta=0 and eta=1)
# -------------------------
def train_flows(config, logs_dir, logger):
    """
    Trains two Flow models: eta=0 and eta=1 using your Flow.train method.
    Saves checkpoints into logs_dir (Flow.train will also save to save_path/checkpoints{eta}).
    Returns trained flow0, flow1 and train/test arrays (numpy).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    X, Y, X_test, Y_test, _ = data_reader(config)

    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    X_test = torch.from_numpy(X_test).float()
    Y_test = torch.from_numpy(Y_test).float()
    print(f"Training data size: {X.shape[0]} and {Y.shape[0]}, Test data size: {X_test.shape[0]} and {Y_test.shape[0]}")
    x_dim = X.shape[1]
    y_dim = Y.shape[1]

    # model kwargs
    model_kwargs = dict(
        n_dim_x=x_dim,
        n_dim_y=y_dim,
        width=config["model"]["width"],
        depth=config["model"]["depth"],
        activation=config["model"]["activation"],
        model_type=config["model"]["network_type"],
        device=device,
        prior_to_posterior=config["model"]["prior_to_posterior"],
        OT = config["model"]["OT"],
        ot_method = config["model"]["ot_method"]
    )

    # create flows for eta=0 and eta=1
    if config["model"]["bridge"]:
        logger.info("Constructing Flow(eta=0) and Flow(eta=1)")
        flow0 = Flow_bridge(eta=0, **model_kwargs)
        flow1 = Flow_bridge(eta=1, **model_kwargs)
    else:
        logger.info("Constructing two independent Flows")
        flow0 = Flow(**model_kwargs)
    


    # Move the internal nets to device (Flow already handles .to(device) in its constructor in your version)
    # Ensure save_path used by Flow.train points to logs_dir
    save_path = logs_dir

    # Train flow0
    logger.info("Training flow_eta0...")
    t0 = time.time()
    loss_history0, test_loss_history0, test_loss_moving_avg_list0, flag1, flag2, patience = flow0.train(
        X=X, Y=Y, X_test=X_test, Y_test=Y_test,
        n_iters=config["training"]["n_iters"],
        batch_size=config["training"]["batch_size"],
        lr=config["training"]["lr"],
        save_path=save_path,
        n_test_freq = config["training"]["n_test_freq"],
        save_freq = config["training"]["save_freq"])
    
    t1 = time.time()
    logger.info(f"Finished training flow0 in {t1 - t0:.1f} sec. Saving final checkpoint.")
    torch.save(flow0.net.state_dict(), os.path.join(logs_dir, "final_flow0.pt"))
    #loss history0 plots
    np.save(os.path.join(logs_dir, "loss0.npy"), np.asarray(loss_history0))
    plt.figure(figsize=(8, 4))
    plt.semilogy(loss_history0, label="Train Loss")
    n_test_freq = config["training"]["n_test_freq"]
    test_loss_x = np.arange(n_test_freq, len(test_loss_history0) * n_test_freq + 1, n_test_freq)
    plt.semilogy(test_loss_x, test_loss_history0, label="Test Loss")  # Test loss (every 200 epochs)
    plt.semilogy(test_loss_x, test_loss_moving_avg_list0, label="Test Loss Moving Avg")  # Moving average of test loss
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("flow0 loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(logs_dir, "loss0_plot.png"), dpi=200)
    plt.close()
    # save stoppping info
    with open(os.path.join(logs_dir, "stopping_info.txt"), "w") as f:
        f.write("patience:\n")
        f.write(", ".join(map(str, patience)) + "\n")

        f.write("iterations in which it would stop (0.1%):\n")
        f.write(", ".join(map(str, flag1)) + "\n")

        f.write("iterations in which it would stop (0%):\n")
        f.write(", ".join(map(str, flag2)) + "\n")

    if config["model"]["bridge"]:
        # Train flow1
        logger.info("Training flow_eta1...")
        t0 = time.time()
        loss_history1 = flow1.train(
            X=X, Y=Y,
            n_iters=config["training"]["n_iters"],
            batch_size=config["training"]["batch_size"],
            lr=config["training"]["lr"],
            save_path=save_path
        )
        t1 = time.time()
        logger.info(f"Finished training flow_eta1 in {t1 - t0:.1f} sec. Saving final checkpoint.")
        torch.save(flow1.net.state_dict(), os.path.join(logs_dir, "final_flow1.pt"))
        #loss history1 plots
        np.save(os.path.join(logs_dir, "loss1.npy"), np.asarray(loss_history1))
        plt.figure(figsize=(8, 4))
        plt.plot(loss_history1)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("flow_eta1 loss")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(logs_dir, "loss1_plot.png"), dpi=200)
        plt.close()

    
    logger.info("Saved loss plots and arrays in logs directory.")

    if config["model"]["bridge"]:
        return flow0, flow1
    else:
        return flow0, None


# -------------------------
# Sampling (uses odeint_sampler_eta)
# -------------------------
def undo_normalization(final_samples):
    x1_max = np.load('/project2/aoberai_286/ihlee/Wildfire_Research/recursive_data/lognorm_x1_log_max.npy')
    x2_max = np.load('/project2/aoberai_286/ihlee/Wildfire_Research/recursive_data/lognorm_x2_log_max.npy')
    x3_max = np.load('/project2/aoberai_286/ihlee/Wildfire_Research/recursive_data/lognorm_x3_log_max.npy')

    x1 = np.exp(np.clip(final_samples[:, 0], 0, None) * x1_max) - 1
    x2 = np.exp(np.clip(final_samples[:, 1], 0, None) * x2_max) - 1
    x3 = np.exp(np.clip(final_samples[:, 2], 0, None) * x3_max) - 1

    return np.stack([x1,x2,x3], axis=1)

def sample_and_save(sampling_dir, logs_dir, config, args, logger):
    """
    For each test sample: sample multiple x from the prior/pool and run odeint_sampler_eta.
    Saves samples_final.npy, per-sample flux plots (optionally), and results.txt
    """
    X_prior, Y_eval_samp, normalizing_factors = data_reader_eval(config)
    if config["data"].get("normalized", False):
        [X_mean, X_scale, Y_mean, Y_scale] = normalizing_factors
    Y_eval_samp = torch.from_numpy(Y_eval_samp).float()
    sample_batch_size = config["sampling"]["sample_batch_size"]
    n_steps = config["sampling"]["n_steps"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"sampling device: {device}")
    x_dim = X_prior.shape[1]
    y_dim = Y_eval_samp.shape[1]
    # model kwargs
    model_kwargs = dict(
        n_dim_x=x_dim,
        n_dim_y=y_dim,
        width=config["model"]["width"],
        depth=config["model"]["depth"],
        activation=config["model"]["activation"],
        model_type=config["model"]["network_type"],
        device=device,
        prior_to_posterior=config["model"]["prior_to_posterior"],
        OT = config["model"]["OT"],
        ot_method = config["model"]["ot_method"]
    )

    # create flows for eta=0 and eta=1
    if config["model"]["bridge"]:
        logger.info("Constructing Flow(eta=0) and Flow(eta=1)")
        flow0 = Flow_bridge(eta=0, **model_kwargs)
        flow1 = Flow_bridge(eta=1, **model_kwargs)
    else:
        logger.info("Constructing two independent Flows")
        flow0 = Flow(**model_kwargs)

    ckpt = torch.load(os.path.join(logs_dir, f"checkpoints/ema_{args.checkpoint}.pt"), map_location=device)
    flow0.vel_ema.ema_model.load_state_dict(ckpt)
    flow0.vel_ema.ema_model.eval()
    flow0.net.eval()
    rng = np.random.default_rng(seed=42)

    generated_samples = {}
    for i in range(Y_eval_samp.shape[0]):
        # get y_cond and true x
        y_cond = Y_eval_samp[i].float().to(device)

        if config["model"]["prior_to_posterior"]:
            #load some examples from test_data or logs (we choose from test_data for simplicity)
            idx = rng.choice(np.arange(X_prior.shape[0]), size=sample_batch_size, replace=False)
            x_start = torch.from_numpy(X_prior[idx]).float().to(device)
        else:
            x_start = torch.randn(sample_batch_size, x_dim, device=device).float()

        # run sampler (odeint with eta mixing)
        if config["model"]["bridge"]:
            samples = flow0.odeint_sampler_eta(x_start=x_start, y_cond=y_cond, n_steps=n_steps, eta1_flow=flow1, return_path=False)
        else:
            samples = flow0.odeint_sampler(x_start=x_start, y_cond=y_cond, n_steps=n_steps, return_path=False)

        # if data were normalized to (-1,1) during training, convert back to (0,1) like your earlier code if requested
        if config["data"].get("normalized", False):
            samples_to_save = samples * X_scale + X_mean
        else:
            samples_to_save = samples
        generated_samples[f"sample_{i}"] = samples_to_save.detach().cpu().clone()
    
    torch.save(generated_samples, os.path.join(sampling_dir,"generated_samples.pt"))
    print(f"Saved generated samples for {len(generated_samples)} in {sampling_dir}")
    #generated_samples = torch.load("generated_samples.pt", map_location="cpu")

    #Recursive sampling
    test_data_dir = "/project2/aoberai_286/ihlee/flow_matching/mlp_code/24hr_test_data"
    os.makedirs("/project2/aoberai_286/ihlee/flow_matching/mlp_code/24hr_results/", exist_ok=True)
    for filename in os.listdir(test_data_dir):
        if filename.endswith('.npy'):
            condition_matrix = np.load(f"{test_data_dir}/{filename}")
            condition_matrix = condition_matrix[:, x_dim:]

            num_ensemble_members = 100
            num_timesteps = condition_matrix.shape[0]
            final_samples_matrix = np.zeros([num_timesteps, num_ensemble_members, 3])
            fire_area_matrix = np.zeros([num_timesteps, num_ensemble_members, 3])
            
            for i in range(num_timesteps):
                latents = torch.randn(num_ensemble_members, x_dim)
                conditions = condition_matrix[i,:]
                conditions_expanded = np.tile(conditions, (num_ensemble_members, 1))
                        
                if i == 1:
                    conditions_expanded[:, 0] = np.sum(fire_area_matrix[:, :, 2], axis=0)/19000
                elif i > 1:
                    conditions_expanded[:, 0] = np.sum(fire_area_matrix[:, :, 2], axis=0)/19000
                    conditions_expanded[:, 1] = np.sum(fire_area_matrix[:i-1, :, 2], axis=0)/14000

                conditions_expanded = torch.from_numpy(conditions_expanded).float().to(device)
                
                samples = flow0.odeint_sampler(x_start=latents.to(device), y_cond=conditions_expanded, n_steps=n_steps, return_path=False)
                final_samples = samples.detach().cpu().numpy()

                final_samples_matrix[i, :, :] = final_samples
                # unnormalize final_samples so fire_area_matrix is in physical values
                final_samples = undo_normalization(final_samples)
                fire_area_matrix[i, :, :] = final_samples
                
            np.savez("/project2/aoberai_286/ihlee/flow_matching/mlp_code/24hr_results/" + f'{filename[:-10]}_raw_samples.npz', final_samples_matrix)
            np.savez("/project2/aoberai_286/ihlee/flow_matching/mlp_code/24hr_results/" + f'{filename[:-10]}_fire_area_matrix.npz', fire_area_matrix)

