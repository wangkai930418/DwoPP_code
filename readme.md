# **Positive Pair Distillation Considered Harmful: Continual Meta Metric Learning for Lifelong Object Re-Identification (BMVC 2022)**

Here we afford the code to reproduce the experimental results on Market-1501 dataset for our paper *"Positive Pair Distillation Considered Harmful: Continual Meta Metric Learning for Lifelong Object Re-Identification".*

[Kai Wang](https://scholar.google.com/citations?user=j14vd0wAAAAJ), [Chenshen Wu](https://scholar.google.com/citations?user=FO7GyVwAAAAJ&hl=en), [Andy Bagdanov](https://scholar.google.com/citations?user=_Fk4YUcAAAAJ&hl=en), [Xialei Liu](https://mmcheng.net/xliu/), [Shiqi Yang](https://www.shiqiyang.xyz/), Shangling Jui and [Joost van de Weijer](https://scholar.google.com/citations?user=Gsw2iUEAAAAJ&hl=en)

## download dataset

Market-1501 dataset can be directly downloaded from http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html

After extracting the files, you need to have the following files structure:
```
|-- market1501  
        |-- bounding_box_train  
        |-- bounding_box_test  
        |-- gt_bbox  
        |-- gt_query  
        |-- query  
        |-- readme.txt
```

Or if you have already had the dataset locally, you can create a soft link to it by:

```
ln -s /your/path/to/market1501 ./
```
## Requirements

All python packages in my experimental environment is listed in *requirements.txt*

## Reproducing

You can directly run the bash files as below.

### Market-1501, DwoPP(Our method)

```
bash CL_DwoPP.sh
```

### Market-1501, DwPP

```
bash CL_DwPP.sh
```

### Market-1501, FT

```
bash CL_FT.sh
```

### Market-1501, Joint training

```
bash joint_train_dmml_market.sh
```

## Others
Other datasets cannot be directly downloaded from their websites due to the privacy issue, please contact the datasets authors. If you have any question, do not hesitate to contact me or post an issue.