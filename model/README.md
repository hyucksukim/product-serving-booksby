## Recbole Baseline Usage

1. Atomic file 생성 (reviewerID, itemID, ratings, timestamp from mongoDB)

```
python make_data.py
```

2. Model training

```
python train.py --config_files [common.yaml] --model [model]
```

+ yaml 파일은 [config] 디렉토리 내부에 위치



++ RecVAE 실행 시 오류 해결

```
TypeError: calculate_loss() missing 1 required positional argument: 'encoder_flag'
```

위 오류는, '''/opt/conda/lib/python3.8/site-packages/recbole/trainer/trainer.py''' 에서
```
def calculate_loss(self, interaction, encoder_flag)
-> def calculate_loss(self, interaction, encoder_flag=True)
```
로 바꿔주시면 실행됩니다.
