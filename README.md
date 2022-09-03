# Code Implementation of Anti-Spoofing Models for ASV Frontend

<strong>Dataset</strong>: ASVSpoof2019 LA Partition<br>
<strong>Attack Type</strong>: Logical Access Attack Detection<br>
<strong>Focus</strong>: Countermeasure Protocols For ASV Systems
```
Models Implemented:

1. Res-TSSDNet & Inc-TSSDNet
2. Res-TSSDNet + Multiresolution CNN
3. RawNet 2 ( + Learnable Filters - Sinc Layer)
4. Multiresolution CNN (Lowrank & Depthwise Convolutions)
```

Results Without Quantization
```
Equal Error Rates (EER) Obtained During Experiments:
X1 - 4.71
X2 - 2.87
X3 - 2.26
X4 - 1.95
X5 - 1.42
```
Results With Quantization

```
Equal Error Rates (EER) Obtained During Experiments:
X1 - 3.67
X2 - 2.24
X3 - 1.42
X4 - 1.27
X5 - 0.74
```

Research Papers Referred & Implemented:

• G. Hua, A. B. J. Teoh and H. Zhang, "Towards End-to-End Synthetic Speech Detection," in IEEE Signal Processing Letters, vol. 28, pp. 1265-1269, 2021, doi: 10.1109/LSP.2021.3089437.

• H. Tak, J. Patino, M. Todisco, A. Nautsch, N. Evans and A. Larcher, "End-to-End anti-spoofing with RawNet2," ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 6369-6373, doi: 10.1109/ICASSP39728.2021.9414234.

• D. Gupta and V. Abrol, "Time-Frequency and Geometric Analysis of Task-Dependent Learning in Raw Waveform Based Acoustic Models," ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022, pp. 4323-4327, doi: 10.1109/ICASSP43922.2022.9746577.
