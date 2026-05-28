# main.py
import os
import shutil
import argparse
import yaml
import numpy as np
import torch

from trainer import set_seed, create_logger, ensure_dir, train_flows, sample_and_save
# Example usage:
#python main.py --config config.yml --save_dir ./run0 --train --checkpoint 1 
#python main.py --config config.yml --save_dir ./run0 --sample --checkpoint 1 
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True, help="path to config.yaml")
    p.add_argument("--save_dir", type=str, required=True, help="experiment name (folder under exp/logs and exp/sampling)")
    p.add_argument("--train", action="store_true", help="Train the model")
    p.add_argument("--sample", action="store_true", help="Run sampling after training")
    p.add_argument("--checkpoint", type=int, default=0, help="checkpoint to sample from. 0 means the last checkpoint")
    p.add_argument("--seed", type=int, default=None, help="manual seed")
    p.add_argument("--n_iter", type=int, default=None, help="number of training loop iterations")
    p.add_argument("--ni", action="store_true", help="if provided, it will remove and remake the directory for logs and sampling. Use this if you want to start a fresh experiment. If not provided, it will keep existing logs and sampling results (if any) and add new ones to the same directories.")
    return p.parse_args()

def main():
    args = parse_args()

    # load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # ----------------------------------------
    # Override config.seed nad noise *only if provided*
    # ----------------------------------------
    if args.seed is not None:
        config["seed"] = args.seed

    if args.n_iter is not None:
        config["training"]["n_iters"] = args.n_iter
    # ----------------------------------------

    # create experiment directories
    seed = config["seed"]
    save_dir = args.save_dir
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if config["model"]["prior_to_posterior"]:
        root_logs = os.path.join(BASE_DIR, "exp", "logs", "p_to_p", save_dir)
        root_sampling = os.path.join(BASE_DIR, "exp", "sampling", "p_to_p", save_dir,  f"ch_{args.checkpoint}")
    else:
        root_logs = os.path.join(BASE_DIR, "exp", "logs", "Gaussian", save_dir)
        root_sampling = os.path.join(BASE_DIR, "exp", "sampling", "Gaussian", save_dir, f"ch_{args.checkpoint}")


    if args.ni:
        if args.train:
            if os.path.exists(root_logs):
                shutil.rmtree(root_logs)
        if args.sample:
            if os.path.exists(root_sampling):
                shutil.rmtree(root_sampling)
    os.makedirs(root_logs, exist_ok=True)
    os.makedirs(root_sampling, exist_ok=True)

    # copy config for reproducibility
    shutil.copy(args.config, os.path.join(root_logs, "config.yaml"))

    # setup logger
    log_file = os.path.join(root_logs, "train.log")
    logger = create_logger(log_file)
    logger.info("Starting experiment")
    logger.info(f"Logs dir: {root_logs}")
    logger.info(f"Sampling dir: {root_sampling}")

    # set seed
    set_seed(seed)
    logger.info(f"Seed set to {seed}")

    # Train flows (this calls flow.train internally)
    if args.train:
        flow0, flow1 = train_flows(config, root_logs, logger)

    # Sampling step
    if args.sample:
        sample_and_save(root_sampling, root_logs, config, args, logger)
        #logger.info(f"Experiment finished. Average error: {ave_error} | Average absolute error: {abs_error}| Average std: {ave_std}")

if __name__ == "__main__":
    main()
