from cmath import exp

from operator import getitem

from turtle import pd

from fastapi import FastAPI, Query

from fastapi.middleware.cors import CORSMiddleware

from typing import List, Optional

from pydantic import BaseModel

import pymongo # import

from pymongo import MongoClient

from bson.json_util import loads, dumps



import requests

import json

import urllib



import pandas as pd

import numpy as np

import random

import time

import torch
import argparse
from recbole.quick_start import load_data_and_model


origins = ["*"]



app = FastAPI()



mongodb_client = "mongodb://118.67.143.144:30001/"



app.add_middleware(

    CORSMiddleware,

    allow_origins=origins,

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', '-m', type=str, default='saved/MacridVAE-cpu_final.pth', help='name of models')
# python run_inference.py --model_path=/opt/ml/input/RecBole/saved/SASRecF-Apr-07-2022_03-17-16.pth 로 실행
args, _ = parser.parse_known_args()

if torch.cuda.is_available():
    args.cuda = True
else:
    args.cuda = False
device = torch.device("cuda" if args.cuda else "cpu")
# model, dataset 불러오기

config, model, dataset, train_data, valid_data, test_data = load_data_and_model(args.model_path)


def api_load_image(asin):

    input_text = asin

    client_id = "EZKJjFYCeCFKfj3qk2OM"

    client_secret = "dLMxOkCV6M"

    input_text_n = urllib.parse.quote(input_text)

    url = "https://openapi.naver.com/v1/search/book?query=" + input_text_n +"&display=3&sort=count"

    Send_request = urllib.request.Request(url)

    Send_request.add_header("X-Naver-Client-Id", client_id)

    Send_request.add_header("X-Naver-Client-Secret", client_secret)

    response = urllib.request.urlopen(Send_request)

    success = response.getcode()

    

    try:

        Response = response.read()

        tokens = json.loads(Response.decode('utf-8'))

        return tokens['items'][0]['image']

    except:

        return "https://bookthumb-phinf.pstatic.net/cover/061/154/06115486.jpg?type=m1&udate=20210317"



@app.get("/search/{asin}")

def load_image(asin):

    time.sleep(0.54)

    client = pymongo.MongoClient(mongodb_client)

    # db(=database): "recsys09"의 "amazon" 데이터베이스

    db = client["amazon"]

    # Collection(=table): "amazon" 데이터 베이스의 "books" 테이블

    Collection = db["url"]

    

    query = {'asin': asin}

    cursor = Collection.find_one(filter=query ,projection={'image_url' : True})

    try:

        return cursor['image_url'][0]

    except:

        return api_load_image(asin)



@app.get("/userdata/{user}")

def getuserdata(user):

    time.sleep(0.1)

    client = MongoClient(mongodb_client)

    db = client["amazon"]

    collection = db["train"]

    query = {'reviewerID': user}

    cursor = collection.find(query, projection={'_id': 0, 'asin':1, 'reviewText':1, 'overall':1, 'title':1, 'startdate':1, 'enddate':1})

    result = loads(dumps(cursor))

    return result



@app.get("/inference/{user}")
def getpredict(user):
    model.eval()
    new_target = user
    
    client = MongoClient(mongodb_client)
    db = client["amazon"]
    collection = db["train"]
    query = {'reviewerID': user}
    
    cursor = collection.find(query, projection={'_id': 0, 'asin':1, 'overall':1})
    result = loads(dumps(cursor))
    asin = [i['asin'] for i in result]
    overall = [((int(i['overall']) - 1) / 4) for i in result]
    
    item_list = []
    for a in asin:
        try:
            item_list.append(dataset.token2id(dataset.iid_field, a))
        except: 
            del overall[asin.index(a)]

    item_list = np.array(item_list)
    top_k = 10
    
    rating_matrix = torch.zeros(1).to(device).repeat(1, dataset.item_num)
    
    row_indices = torch.zeros(1).to(device).repeat(len(item_list)).type(torch.int64)

    col_indices = torch.from_numpy(item_list).type(torch.LongTensor)
    
    item_values = torch.Tensor(overall).to(device)
    
    rating_matrix.index_put_((row_indices, col_indices), item_values)
    
    score, _, _ = model.forward(rating_matrix)
    
    prediction = torch.topk(score, top_k).indices
    
    prediction = dataset.id2token(dataset.iid_field, prediction.cpu())[0]

    return list(prediction)
    

li = []

class Inter(BaseModel):

    id: str

    title: str

    asin: str

    rate: int

    review: str

    startdate: str

    enddate: str

    

@app.post("/intersave")

def intersave(inter:Inter):

    

    client = pymongo.MongoClient(mongodb_client)

    db = client["amazon"]

    collection = db["train"]

    if inter.review:

        query = {'asin': inter.asin, 'title':inter.title, 'reviewerID': inter.id, 'overall': inter.rate, 'reviewText': inter.review, 'startdate':inter.startdate, 'enddate':inter.enddate}

    else:

        query = {'asin': inter.asin, 'title':inter.title, 'reviewerID': inter.id, 'overall': inter.rate, 'startdate':inter.startdate, 'enddate':inter.enddate}

    try:

        cursor = collection.insert_one(query)

        return "done"

    except:

        return "fail"



@app.get("/items/{asin}")

def get_item(asin):

    time.sleep(0.54)

    client = MongoClient(mongodb_client)

    db = client["amazon"]

    collection = db["books"]

    query = {'asin': asin}

    cursor = collection.find(query, projection={'_id': False})

    result = loads(dumps(cursor))

    return result[0]



@app.get("/title/{asin}")

def get_title(asin):

    items = get_item(asin)

    return items[0]['title']





def quick_img(asin):

    client = pymongo.MongoClient(mongodb_client)

    # db(=database): "recsys09"의 "amazon" 데이터베이스

    db = client["amazon"]

    # Collection(=table): "amazon" 데이터 베이스의 "books" 테이블

    Collection = db["url"]

    

    query = {'asin': asin}

    cursor = Collection.find_one(filter=query ,projection={'image_url' : True})

    try:

        return cursor['image_url'][0]

    except:

        return api_load_image(asin)



@app.get("/itemsli/{item}")

def get_itemsli(item):

    client = pymongo.MongoClient("mongodb://118.67.143.144:30001/")



    db = client["amazon"]

    collection = db["books"]



    search_word = item



    query = {"title": {"$regex": search_word, "$options": "i"}}



    cursor = collection.find(filter=query, projection={"_id": False, "asin": True, "title":True, "category":True, "description":True, "price":True}).skip(0).limit(10)



    li = []

    for result in cursor:

        tmp = result

        tmp.update({"imgurl":quick_img(result['asin'])})

        li.append(tmp)

    return li



@app.post("/del")

def item_delete(inter:Inter):

    client = pymongo.MongoClient("mongodb://118.67.143.144:30001/")

    db = client["amazon"]

    collection = db["train"]

    query = {'asin': inter.asin, 'title':inter.title, 'reviewerID': inter.id, 'overall': inter.rate, 'reviewText': inter.review, 'startdate':inter.startdate, 'enddate':inter.enddate}

    cursor = collection.delete_one(query)



@app.get("/delete/{user}")

def fsafa(user):

    client = pymongo.MongoClient("mongodb://118.67.143.144:30001/")

    db = client["amazon"]

    collection = db["train"]

    query = {"user":user}

    cursor = collection.delete_many(query)


