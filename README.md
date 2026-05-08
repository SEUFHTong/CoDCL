## Overview

## 1. Benchmark Datasets and Preprocessing
We can run ```preprocess_data/preprocess_data.py``` for pre-processing the datasets.
For example, to preprocess the *Wikipedia* dataset, we can run the following commands:
```{bash}
cd preprocess_data/
python preprocess_data.py  --dataset_name wikipedia
```
We can also run the following commands to preprocess all the original datasets at once:
```{bash}
cd preprocess_data/
python preprocess_all_data.py
```
In order to save reviewers' time, we will make the counterfactual link completion results of a single run of "Counterfactual link completion" public.
which can be downloaded https://drive.google.com/drive/folders/1UnSWULnKLHuWRBEKwuHeLY4rxUpPMUYI?usp=sharing
Please download them and put them in```\processed_data\my_dataset_folder```folder.

##  2.Train and Evaluation 

We've pre-configured the run code in ```run.sh``` for one-click training. Alternatively, you can run:

```python main.py --dataset_name uci --model_name DyGFormer --load_best_configs --num_runs 5 --gpu 0```  for different datasets.

We support dynamic link prediction in both conductive and inductive settings.

To save time, we've stored the trained weights in the ```saved_models/DyGFormer``` directory; you only need to load them later.



### 3.Scripts for Dynamic Link Prediction
If you want to load the best model configurations determined by the grid search, please set the *load_best_configs* argument to True.
