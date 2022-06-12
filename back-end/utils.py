import pandas as pd
import numpy as np
import os

from recommenders.datasets.split_utils import min_rating_filter_pandas
from recommenders.datasets.sparse import AffinityMatrix
from recommenders.datasets.python_splitters import numpy_stratified_split
from recommenders.utils.python_utils import binarize


class Data:
    def __init__(self, config):
        self.config = config

        self.data_path = os.path.join(config["root_dir"], "data", config["data_name"])

        self.df = self.load_parquet_file(self.data_path)
        self.df_preferred, self.df_low_rating = self.binarize_data(
            config["threshhold"], self.df
        )
        self.df = min_rating_filter_pandas(
            self.df_preferred, min_rating=config["user_min_rating"], filter_by="user"
        )
        self.df = min_rating_filter_pandas(
            self.df, min_rating=config["item_min_rating"], filter_by="item"
        )
        self.df.to_csv("books.csv", index=False)
        
        self.create_train_valid_test_users()
        self.get_data_set()
        self.unique_train_items = self.get_unique_train_items()
        unique_train_items_path = os.path.join(config["root_dir"], "data", "train_items.npy")
        np.save(unique_train_items_path, self.unique_train_items)
        self.create_val_test_set()
        self.create_matrix()
        self.split_dataset()

    def load_parquet_file(self, data_path: str) -> pd.DataFrame:
        df = pd.read_parquet(data_path, engine="pyarrow")
        df = df[["reviewerID", "asin", "overall", "unixReviewTime"]]
        df.columns = ["userID", "itemID", "rating", "timestamp"]
        return df

    def binarize_data(self, threshhold: float, df: pd.DataFrame) -> pd.DataFrame:
        df_preferred = df[df["rating"] > threshhold]
        df_low_rating = df[df["rating"] <= threshhold]
        return df_preferred, df_low_rating

    def _get_unique_users(self) -> np.ndarray:
        unique_users = sorted(self.df.userID.unique())
        np.random.seed(self.config["seed"])
        unique_users = np.random.permutation(unique_users)
        return unique_users

    def create_train_valid_test_users(self) -> None:
        self.unique_users = self._get_unique_users()
        self.n_users = len(self.unique_users)
        self.train_users = self.unique_users[
            : (self.n_users - self.config["heldout_users"] * 2)
        ]
        self.val_users = self.unique_users[
            (self.n_users - self.config["heldout_users"] * 2) : (
                self.n_users - self.config["heldout_users"]
            )
        ]
        self.test_users = self.unique_users[
            (self.n_users - self.config["heldout_users"]) :
        ]

    def get_data_set(self) -> None:
        self.train_set = self.df.loc[self.df["userID"].isin(self.train_users)]
        self.val_set = self.df.loc[self.df["userID"].isin(self.val_users)]
        self.test_set = self.df.loc[self.df["userID"].isin(self.test_users)]

    def get_unique_train_items(self) -> np.ndarray:
        return pd.unique(self.train_set["itemID"])

    def create_val_test_set(self) -> None:
        # For validation set keep only movies that used in training set
        self.val_set = self.val_set.loc[
            self.val_set["itemID"].isin(self.unique_train_items)
        ]
        # For test set keep only movies that used in training set
        self.test_set = self.test_set.loc[
            self.test_set["itemID"].isin(self.unique_train_items)
        ]

    def create_matrix(self) -> None:
        self.am_train = AffinityMatrix(
            df=self.train_set, items_list=self.unique_train_items
        )
        self.am_val = AffinityMatrix(
            df=self.val_set, items_list=self.unique_train_items
        )
        self.am_test = AffinityMatrix(
            df=self.test_set, items_list=self.unique_train_items
        )

        self.train_data, _, _ = self.am_train.gen_affinity_matrix()
        (
            self.val_data,
            self.val_map_users,
            self.val_map_items,
        ) = self.am_val.gen_affinity_matrix()
        (
            self.test_data,
            self.test_map_users,
            self.test_map_items,
        ) = self.am_test.gen_affinity_matrix()

    def split_dataset(self) -> None:
        self.val_data_tr, self.val_data_te = numpy_stratified_split(
            self.val_data, ratio=self.config["ratio"], seed=self.config["seed"]
        )
        self.test_data_tr, self.test_data_te = numpy_stratified_split(
            self.test_data, ratio=self.config["ratio"], seed=self.config["seed"]
        )
        self.train_data = binarize(a=self.train_data, threshold=3.5)
        self.val_data = binarize(a=self.val_data, threshold=3.5)
        self.test_data = binarize(a=self.test_data, threshold=3.5)

        # Binarize validation data: training part
        self.val_data_tr = binarize(a=self.val_data_tr, threshold=3.5)

        # Binarize validation data: testing part (save non-binary version in the separate object, will be used for calculating NDCG)
        self.val_data_te_ratings = self.val_data_te.copy()
        self.val_data_te = binarize(a=self.val_data_te, threshold=3.5)

        # Binarize test data: training part
        self.test_data_tr = binarize(a=self.test_data_tr, threshold=3.5)

        # Binarize test data: testing part (save non-binary version in the separate object, will be used for calculating NDCG)
        self.test_data_te_ratings = self.test_data_te.copy()
        self.test_data_te = binarize(a=self.test_data_te, threshold=3.5)

        # retrieve real ratings from initial dataset

        self.test_data_te_ratings = pd.DataFrame(self.test_data_te_ratings)
        self.val_data_te_ratings = pd.DataFrame(self.val_data_te_ratings)

        for index, i in self.df_low_rating.iterrows():
            user_old = i["userID"]  # old value
            item_old = i["itemID"]  # old value

            if (self.test_map_users.get(user_old) is not None) and (
                self.test_map_items.get(item_old) is not None
            ):
                user_new = self.test_map_users.get(user_old)  # new value
                item_new = self.test_map_items.get(item_old)  # new value
                rating = i["rating"]
                self.test_data_te_ratings.at[user_new, item_new] = rating

            if (self.val_map_users.get(user_old) is not None) and (
                self.val_map_items.get(item_old) is not None
            ):
                user_new = self.val_map_users.get(user_old)  # new value
                item_new = self.val_map_items.get(item_old)  # new value
                rating = i["rating"]
                self.val_data_te_ratings.at[user_new, item_new] = rating

        self.val_data_te_ratings = self.val_data_te_ratings.to_numpy()
        self.test_data_te_ratings = self.test_data_te_ratings.to_numpy()
