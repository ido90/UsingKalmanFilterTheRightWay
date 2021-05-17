# Noise Estimation Is Not Optimal: How to Use Kalman Filter the Right Way

This repo implements the experiments for the paper [Noise Estimation Is Not Optimal: How To Use Kalman Filter The Right Way](https://arxiv.org/abs/2104.02372) by Ido Greenberg, Shie Mannor and Netanel Yannay.

### Dependencies and instructions
Dependencies: pytorch, numpy, pandas, matplotlib, seaborn.

Before the training the following directories should be created: `output/`, `data/models`, `data/train`, `data/XY`. If not created in advance, certain saving actions may fail and stop the whole training in the middle.

Using the tools in this repo or reconstructing the results of the paper can be done by following the corresponding notebooks:
* `Neural Kalman Filter.ipynb`: Neural KF (Section 3 in the paper) - training + evaluation + analysis.
* `Optimized Kalman Filter.ipynb`: Optimized KF case study (Section 4) - training + evaluation + analysis.
* `Paper Figures.ipynb`: Re-generation of exact figures from the paper.
* `OKF Training Example.ipynb`: A simple training example.
* `Missing Detections.ipynb`: An example for training with missing detections.

Note: since the anonymized version of the repo omits some of the notebooks, they are also available [here](https://drive.google.com/drive/folders/1I3rgOCxxzVg6lsIZB7EKl1WAi3cNSE-N?usp=sharing).

### Repo contents
* A **simulation of aerial targets** (`ScenarioSimulator.py`, `TargetSimulator.py`, `BasicTargetSimulator.py`, `TargetSimulator.ipynb`) and a **Doppler radar** (`SensorGenerator.py`, `RadarSimulator.ipynb`).
* Implementation of **recurrent predictive models for tracking**, following the framework of Kalman Filter (`Trackers.py`, `NeuralTrackers.py`).
* **Optimization** and analysis of the tracking models (`PredictionLab.py`).
* Simulation and analysis of **multi-target tracking** (`TrackingLab.py`).
* Implementation of models and tests for the additional domain of **tracking from video** (`MOT20/`, `OKF_MOT20.ipynb`).
* Notebooks with **further results and examples** (see above).
