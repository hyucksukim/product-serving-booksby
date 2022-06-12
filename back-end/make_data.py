import pymongo
import pandas as pd
import os

# db에서 데이터 불러오기

client = pymongo.MongoClient("mongodb://118.67.143.144:30001/")

db = client["amazon"]
collection = db["train"]

cursor = collection.find(
    {},
    projection={
        "_id": False,
        "reviewerID": True,
        "asin": True,
        "overall": True,
        "unixReviewTime": True,
    },
)

df = pd.DataFrame.from_dict(cursor)
df.columns = ["userID", "itemID", "rating", "timestamp"]


# .inter file 생성

TARGET_DIR = os.path.join(os.getcwd(), "../data/boostcamp")
TARGET_NAME = "boostcamp.inter"

os.makedirs(TARGET_DIR, exist_ok=True)

df = df.rename(
    columns={
        "userID": "user_id:token",
        "itemID": "item_id:token",
        "rating": "rating:float",
        "timestamp": "timestamp:float",
    }
)
df.to_csv(os.path.join(TARGET_DIR, TARGET_NAME), index=False, sep="\t")

print("Done!")
