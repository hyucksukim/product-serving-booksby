import argparse
import yaml
import os

from recommenders.models.vae.multinomial_vae import Mult_VAE
from recommenders.utils.timer import Timer
import tensorflow._api.v2.compat.v1 as tf2

from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="name of models")
    parser.add_argument("--dataset", type=str, default=None, help="name of datasets")
    parser.add_argument("--config_file", type=str, default=None, help="config files")

    args = parser.parse_args()

    if not args.model:
        raise ValueError("Model is not specified.")
    if not args.config_file:
        raise ValueError("Config file is not specified.")

    if not args.config_file.endswith(".yaml"):
        args.config_file += ".yaml"
    config_file = os.path.join("config", args.config_file)
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

    tf2.disable_v2_behavior()

    save_path = os.path.join(config["root_dir"], "model", config["weights_name"])
    data_path = os.path.join(config["root_dir"], config["data_name"])

    data = Data(config)

    model_without_anneal = Mult_VAE(
        n_users=data.train_data.shape[0],  # Number of unique users in the training set
        original_dim=data.train_data.shape[1],  # Number of unique items in the training set
        intermediate_dim=config["intermediate_dim"],
        latent_dim=config["latent_dim"],
        n_epochs=config["epochs"],
        batch_size=config["batch_size"],
        k=config["top_k"],
        verbose=1,
        seed=config["seed"],
        save_path=save_path,
        drop_encoder=0.5,
        drop_decoder=0.5,
        annealing=False,
        beta=1.0,
    )

    with Timer() as t:
        model_without_anneal.fit(
            x_train=data.train_data,
            x_valid=data.val_data,
            x_val_tr=data.val_data_tr,
            x_val_te=data.val_data_te_ratings,
            mapper=data.am_val,
        )

    print("Took {} seconds for training.".format(t))
