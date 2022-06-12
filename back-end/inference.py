import pandas as pd
import numpy as np
import os
import yaml

from utils import *

from recommenders.models.vae.multinomial_vae import Mult_VAE
from recommenders.datasets.sparse import AffinityMatrix
import tensorflow._api.v2.compat.v1 as tf2


def load_model():
    config_file = "./config/config.yaml"
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tf2.disable_v2_behavior()

    save_path = os.path.join(config["root_dir"], "model", config["weights_name"])
    data_path = os.path.join(config["root_dir"], config["data_name"])

    # data = Data(config)
    # print(data.train_data.shape[0])
    # print(data.train_data.shape[1])

    model = Mult_VAE(
        # n_users=data.train_data.shape[0],  # Number of unique users in the training set
        # original_dim=data.train_data.shape[1],  # Number of unique items in the training set
        n_users=13097,
        original_dim=63750,
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
    model_path = os.path.join(config['root_dir'], 'model', config['weights_name'])
    model.model.load_weights(model_path)

    return model
