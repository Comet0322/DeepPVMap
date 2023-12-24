# DeepPVMap
## Introduction
This is an open source code of paper **DeepPVMap: Deep Photovoltaic Map for Efficient Segmentation of Solar Panels from Low-Resolution Aerial Imagery**. 

## Environment compatibilities

The project was developed using the following environments.

| Env | versions |
| --- | --- |
| os  | `ubuntu-20.04` |
| python | `3.10` |
| pytorch | `1.13.1` |

**Install**

Install the required dependencies:
```
pip install -r requirements.txt
```
**Usage**

To train the model, run the following command:
```
bash ./run.sh <export directory> <data directory> <experiment name>
```

<mark>The map generation script is under development.</mark>

**Result**

Photovoltaic (PV) panels predicted using orthophotos from Taiwan National Land Surveying and Cartography Center (NLSC), collected in June 2022.

<img src="https://github.com/Comet0322/DeepPVMap/assets/89444006/9555d7f1-578c-4c4e-947e-9e6a758ac1c8" alt="pred_pv_distribution" width="400"/>

<img src="https://github.com/Comet0322/DeepPVMap/assets/89444006/6dde8df6-e601-4487-874a-2c3361228cd3" alt="area_heatmap" width="400"/>
