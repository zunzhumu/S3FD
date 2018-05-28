two stages strategies\
stage one: each gt matched anchor number:7\
stage one: each gt matched anchor number:2\
stage one: each gt matched anchor number:1\
stage one: each gt matched anchor number:1\
stage one: each gt matched anchor number:1\
stage one: each gt matched anchor number:3\
stage one: each gt matched anchor number:1\
stage one: each gt matched anchor number:9\
stage one: each gt matched anchor number:7\
stage one: each gt matched anchor number:9\
stage one: each gt matched anchor number:14\
the ground truth number:11\
the averge anchors matched number:5\
deal with tiny and outer face\
stage two: each gt matched anchor number:7\
stage two: each gt matched anchor number:5\
stage two: each gt matched anchor number:5\
stage two: each gt matched anchor number:5\
stage two: each gt matched anchor number:5\
stage two: each gt matched anchor number:5\
stage two: each gt matched anchor number:5\
stage two: each gt matched anchor number:9\
stage two: each gt matched anchor number:7\
stage two: each gt matched anchor number:9\
stage two: each gt matched anchor number:14
# S3FD: Single Shot Scale-invariant Face Detector

### Getting started
* You will need python modules: `cv2`, `matplotlib` and `numpy`.
If you use mxnet-python api, you probably have already got them.
You can install them via pip or package managers, such as `apt-get`:
```
sudo apt-get install python-opencv python-matplotlib python-numpy
```
## Note The scale compensation anchor matching strategy is written into multibox_target.cu
* Copy multibox_target_operator/multibox_target.cc multibox_target.cu to mxnet/src/operator/contrib to cover original multibox_target.cc multibox_target.cu

* Build MXNet: Follow the official instructions
```

### Train the model
This example only covers training on Wider Face dataset. Other datasets should
* Download the converted pretrained `vgg16_reduced` model [here](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.2-beta/vgg16_reduced.zip), unzip `.param` and `.json` files
into `model/` directory by default.
* Download the Wider Face dataset, skip this step if you already have one.
* Extra data in data/widerface/WIDER_train WIDER_val wider_face_split
* Convert voc format: 
cd data/widerface
python widerface_voc.py
* Convert .rec data
python tools/prepare_dataset.py --dataset widerface --set train --target ./data/train.lst
python tools/prepare_dataset.py --dataset widerface --set val --target ./data/val.lst --shuffle False
```
* Start training:
```
python train.py

### NOTE!!!!!!!!
### By default,this example use data_shape=608, if you have enough GPU memory, you should set data_shape=640.
I only have one GTX1080, so I don't have enough time to train.I use 0.001 learning rate for 7 epochs, the mAP achieved 64% in all validation set.You can try it at will.

