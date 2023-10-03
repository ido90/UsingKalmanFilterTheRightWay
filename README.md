# Optimization or Architecture: How to Hack Kalman Filtering

This repo implements the experiments for the paper [Optimization or Architecture: How to Hack Kalman Filtering](https://arxiv.org/abs/2310.00675) by Ido Greenberg, Netanel Yannay and Shie Mannor.

See a separate repo for the [PyPI package](https://pypi.org/project/Optimized-Kalman-Filter/) of the Optimized KF (OKF).

<img src="https://github.com/ido90/UsingKalmanFilterTheRightWay/blob/master/poster/Poster.png" width="960">

### Dependencies and instructions
Dependencies: pytorch, numpy, pandas, matplotlib, seaborn.

Before the training the following directories should be created: `./output/`, `./data/models`, `./data/train`, `./data/XY`. If not created in advance, certain saving actions may fail and stop the whole training in the middle.

Using the tools in this repo or reconstructing the results of the paper can be done by following the corresponding notebooks:
* `Optimized Kalman Filter.ipynb`: Optimized KF case study (Section 5.1 and Appendix B in the paper) - data generation + training + evaluation + analysis.
* `Neural Kalman Filter.ipynb`: Neural KF (Section 4 and Appendix C) - data generation + training + evaluation + analysis.
* `OKF_MOT20.ipynb` (Section 5.2) and `OKF_lidar.ipynb` (Section 5.3) - data generation + training + evaluation + analysis.
* `Paper Figures.ipynb`: Re-generation of certain figures from the paper.
* `OKF Training Example.ipynb`: A simple training example.
* `Missing Detections.ipynb`: An example for training with missing detections.

### Repo contents
* A compact implementation of **Optimized Kalman Filter** (`OKF.py`) - used for the Lidar and MOT20 experiments.
* A **simulation of aerial targets** (`ScenarioSimulator.py`, `TargetSimulator.py`, `BasicTargetSimulator.py`, `TargetSimulator.ipynb`) and a **Doppler radar** (`SensorGenerator.py`, `RadarSimulator.ipynb`).
* Implementation of **recurrent predictive models for tracking**, following the framework of Kalman Filter (`Trackers.py`, `NeuralTrackers.py`).
* **Optimization** and analysis of the tracking models (`PredictionLab.py`).
* Simulation and analysis of **multi-target tracking** (`TrackingLab.py`).
* Implementation of models and tests for the additional domains of **tracking from video** (`MOT20/`, `OKF_MOT20.ipynb`) and **self-driving state-estimation from lidar measurements** (`Driving/`, `OKF_lidar.ipynb`).
* Notebooks with **further results and examples** (see above).

### Cite us
```
@article{greenberg2023okf,
  title={Optimization or architecture: how to hack Kalman filtering},
  author={Greenberg, Ido and Yannay, Netanel and Mannor, Shie},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```
