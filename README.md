## train
if you use GPU,
```
$ CUDA_VISIBLE_DEVICES=0 python train.py --gpu True --data /path/to/dataset --save /path/to/save-model
```
else,
```
python train.py -data /path/to/dataset --save /path/to/save-model
```

## predict
```
$ python predict.py
```

## change format to submit
```
$ python submit.py
```
