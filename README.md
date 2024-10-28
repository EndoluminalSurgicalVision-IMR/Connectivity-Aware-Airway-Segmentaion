# Connectivity-Aware-Airway-Segmentaion

[**_Towards Connectivity-Aware Pulmonary Airway Segmentation_**](https://ieeexplore.ieee.org/document/10283811)

> By Minghui Zhang, Yun Gu
>> Institute of Medical Robotics, Shanghai Jiao Tong University, Shanghai, China

>> Institute of Image Processing and Pattern Recognition, Shanghai Jiao Tong University, Shanghai, China


## News

[2024/10/23] 🏆🥈🥉 The proposed model won several prizes in different tracks of [**MICCAI 2024 TopCoW Challenge**](https://topcow24.grand-challenge.org/). We uploaded the solution in the ```challenge_solution/TopCow24_Challenge```.

[2024/10/23] 🌟 The proposed model achieved the Honorable Methion in the [**MICCAI 2024 Aorta Challenge**](https://aortaseg24.grand-challenge.org/). We uploaded the solution in the ```challenge_solution/Aorta24_Challenge```.

[2023/10/12] 🥈 The proposed model won the first Runner-up in the [**MICCAI 2023 AIIB Challenge**](https://codalab.lisn.upsaclay.fr/competitions/13238). We will upload the solution in the ```challenge_solution/AIIB23_Challenge```.


## Introduction
Detailed pulmonary airway segmentation is a clinically important task for endobronchial intervention and treatment of peripheral pulmonary lesions. 
Breakage of small bronchi distals cannot be effectively eliminated in the prediction results of CNNs, which is detrimental to use as a reference for bronchoscopic-assisted surgery. 
We proposed a connectivity-aware segmentation method to improve the performance of airway segmentation. 
A Connectivity-Aware Surrogate (CAS) module is first proposed to balance the training progress within-class distribution. 
Furthermore, a Local-Sensitive Distance (LSD) module is designed to identify the breakage and minimize the variation of the distance map between the prediction and ground-truth.

## Usage
To quick start, we provided the pretained networks, and can try the script in ```tests/_test_airway_model```

```
python _test_airway_model.py
```

You can download our pretrained checkpoint from [here](https://drive.google.com/file/d/1_Uz2DzVHAa0S1fRNqyfYRYQZQixbO_dT/view?usp=sharing). The configs and models are specified in 
```configs/airway_config``` and ```networks/airway_network```.

The implementation of two modules CAS and LSD is modularized in the ```networks/CAS```,```networks/LSD```.

## 📝 Citation
If you find this repository or our paper useful, please consider citing our paper:
```
@article{zhang2023towards,
  title={Towards Connectivity-Aware Pulmonary Airway Segmentation},
  author={Zhang, Minghui and Gu, Yun},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2023},
  publisher={IEEE}
}
```
