# S2F2: Single-Stage Flow Forecasting for Future Multiple Trajectories Prediction

Detection, tracking, and forecasting from raw images in an end-to-end network. 

Anonymous github for ECCV2022.

![image](https://github.com/carrieeeeewithfivee/S2F2/blob/master/combine.gif)

## Getting Started
### Installing

* Tested on Cuda10, Python 3.7.10, Pytorch 1.2.0
* We use MOT17 and MOT20 as training and validation data. Refer to the official webpage of MOT challenge to download the data. After downloading, you should prepare the data in the following structure:
```
MOT17
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train(empty)
MOT20
   |——————images
   |        └——————train
   |        └——————test
   └——————labels_with_ids
            └——————train(empty)
```
Then, you can run src/gen_data/gen_labels_17.py and src/gen_data/gen_labels_20.py to generate the labels_with_ids of MOT17 and MOT20.
* You need to change the data path in src/lib/opts.py under the --data_dir arg, and src/lib/config files.
* We use DCNv2 in our backbone network. DCNv2 can be cloned at [Link](https://github.com/CharlesShang/DCNv2).

### Models
* We use DLA-34 COCO pretrained model: [DLA-34 official](https://drive.google.com/file/d/1pl_-ael8wERdUREEnaIfqOV_VF2bEVRT/view)
* Our trained models can be downloaded at [Link](https://drive.google.com/drive/folders/1b7CyxfRG6HPOFVlMt0LHPN2QUHrQH7Db?usp=sharing)
* After downloading, you should put the pretrained models in the following structure:
```
${ROOT}
   └——————models
           └——————ctdet_coco_dla_2x.pth
```

### Run Training

* After downloading the dataset, run:
```
sh experiments/forecast_staticMOT.sh 
```
to start training staticMOT, or
```
sh experiments/forecast_mot17.sh 
```
to train on mot17. Change the .sh files for different settings.
Validation commands are also written in the .sh files. Validatation datasets need to be generated according to S2F2/src/data/MOT_static_val_half and S2F2/src/data/MOT17_val_half/. Parsing files directly has not been implemented yet. 
