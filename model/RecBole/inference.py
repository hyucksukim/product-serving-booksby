import argparse
import torch
from recbole.quick_start import load_data_and_model


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, default='saved/MacridVAE-Jun-09-2022_08-42-00.pth', help='name of models')
    
    # python run_inference.py --model_path=/opt/ml/input/RecBole/saved/SASRecF-Apr-07-2022_03-17-16.pth 로 실행
    
    args, _ = parser.parse_known_args()

    device = torch.device("cuda")

    # model, dataset 불러오기
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(args.model_path)

    model.eval()

    new_target = 'lck'

    asin_list = ['0001720392', '0007173156', '0030845742', '0060248025']

    item_list = dataset.token2id(dataset.iid_field, asin_list)

    top_k = 10

    rating_matrix = torch.zeros(1).to(device).repeat(1, dataset.item_num)

    
    row_indices = torch.zeros(1).to(device).repeat(len(item_list)).type(torch.int64)
    col_indices = torch.from_numpy(item_list)
    item_values = torch.ones(1).to(device).repeat(len(item_list))

    rating_matrix.index_put_((row_indices, col_indices), item_values)

    score, _, _ = model.forward(rating_matrix)

    prediction = torch.topk(score, top_k).indices

    prediction = dataset.id2token(dataset.iid_field, prediction.cpu())[0]

    print(prediction)