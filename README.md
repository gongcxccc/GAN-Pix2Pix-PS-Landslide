## 必看前提

要将图像先调整成1024×512尺寸的。数据集中每张图像都要这样才行。
无论是训练集还是测试集，都要保持一致。那么这里推荐都是1024×512。

## 训练

```python
python train.py --name dataset_pix2pix --label_nc 0 --no_instance --gpu_ids 0 
```

==不执行随机翻转操作==

```python
python train.py --name dataset_pix2pix --label_nc 0 --no_instance --gpu_ids 0  --no_flip
```

==不执行随机翻转，不打乱数据集顺序。==

```python
python train.py --name dataset_1111 --label_nc 0 --no_instance --gpu_ids 0  --no_flip --serial_batches
```

==不执行随机翻转，不打乱数据集顺序。设定训练次数==

```python
python train.py --name dataset_pix2pix --label_nc 0 --no_instance --gpu_ids 0  --no_flip --serial_batches --niter xxx --niter_decay xxx
```
```python
python train.py --name dataset_pix2pix --label_nc 0 --no_instance --gpu_ids 0 --niter xxx --niter_decay xxx
```
```python
python train.py --name dataset_pix2pix --label_nc 0 --no_instance --gpu_ids 0 --niter xxx --niter_decay xxx
```
```python
python train.py --name dataset_pix2pix --label_nc 0 --no_instance --gpu_ids 0 --niter xxx --niter_decayxxx
```
## 测试

```python
python test.py --name dataset_pix2pix --ngf xxx --label_nc 0 --no_instance --how_many xxx
```

