## train
If you use GPU,
```
$ CUDA_VISIBLE_DEVICES=0 python train.py --gpu True --data /path/to/dataset/ --save /path/to/save_model/
```
else,
```
$ python train.py --data /path/to/dataset/ --save /path/to/save-model/
```

## predict
```
$ python predict.py --model /path/to/trained_model --test_preprocessed /path/to/preprocessed_test_images/ --test_original /path/to/original_test_images/ --save /path/to/save_output/
```

## change format to submit
```
$ python submit.py
```
