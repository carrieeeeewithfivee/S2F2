# S2F2: Single-Stage Flow Forecasting for Future Multiple Trajectories Prediction

Detection, tracking, and forecasting from raw images in an end-to-end network. 

Anonymous github for ECCV2022.

![image](https://imgur.com/3yHJFI0)

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
Then, you can run src/gen_labels_17.py and src/gen_labels_20.py to generate the labels of MOT17 and MOT20.
* You need to change the data path in src/lib/opts.py under the --data_dir arg, and src/lib/config files.

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
Validation commands are also written in the .sh files. Navigate to src/ to validate.
