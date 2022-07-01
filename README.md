# The Fragility of Noise Estimation in Kalman Filter: Optimization Can Handle Model-Misspecification

This repo implements the experiments for the paper [The Fragility of Noise Estimation in Kalman Filter: Optimization Can Handle Model-Misspecification](https://arxiv.org/abs/2104.02372) by Ido Greenberg, Shie Mannor and Netanel Yannay.

For a more compact implementation of the Optimized KF, see the corresponding [PyPI package](https://pypi.org/project/Optimized-Kalman-Filter/).

<img src="https://github.com/ido90/UsingKalmanFilterTheRightWay/blob/master/poster/Poster.png" width="960">

### Dependencies and instructions
Dependencies: pytorch, numpy, pandas, matplotlib, seaborn.

Before the training the following directories should be created: `output/`, `data/models`, `data/train`, `data/XY`. If not created in advance, certain saving actions may fail and stop the whole training in the middle.

Using the tools in this repo or reconstructing the results of the paper can be done by following the corresponding notebooks:
* `Optimized Kalman Filter.ipynb`: Optimized KF case study (Section 4 and Appendices G,H in the paper) - data generation + training + evaluation + analysis.
* `Neural Kalman Filter.ipynb`: Neural KF (Section 5 and Appendix K) - data generation + training + evaluation + analysis.
* `OKF_MOT20.ipynb` (Appendix I) and `OKF_lidar.ipynb` (Appendix J) - data generation + training + evaluation + analysis.
* `Paper Figures.ipynb`: Re-generation of figures from the paper.
* `OKF Training Example.ipynb`: A simple training example.
* `Missing Detections.ipynb`: An example for training with missing detections.

Note: since the anonymized version of the repo omits some of the notebooks, they are also available [here](https://drive.google.com/drive/folders/1I3rgOCxxzVg6lsIZB7EKl1WAi3cNSE-N?usp=sharing).

### Repo contents
* A compact implementation of **Optimized Kalman Filter** (`OKF.py`) - used for the Lidar and MOT20 experiments.
* A **simulation of aerial targets** (`ScenarioSimulator.py`, `TargetSimulator.py`, `BasicTargetSimulator.py`, `TargetSimulator.ipynb`) and a **Doppler radar** (`SensorGenerator.py`, `RadarSimulator.ipynb`).
* Implementation of **recurrent predictive models for tracking**, following the framework of Kalman Filter (`Trackers.py`, `NeuralTrackers.py`).
* **Optimization** and analysis of the tracking models (`PredictionLab.py`).
* Simulation and analysis of **multi-target tracking** (`TrackingLab.py`).
* Implementation of models and tests for the additional domains of **tracking from video** (`MOT20/`, `OKF_MOT20.ipynb`) and **self-driving state-estimation from lidar measurements** (`Driving/`, `OKF_lidar.ipynb`).
* Notebooks with **further results and examples** (see above).
