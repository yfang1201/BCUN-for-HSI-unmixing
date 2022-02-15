# Comparison of Unmixing method

## Envirionment Setup: 

Monte Carlo runs: 5 

Number of endmembers to estimate: 7 

Number of skewers (PPI): 1000 

Maximum number of iterations (N-FINDR): 21 

### Comparison between the ground-truth and extracted endmembers using PPI, N-FINDR, VCA using SAD for the simulate Dataset.

| Endmembers   |   AEsmm |    VCA |    PPI |   NFINDR |   KPmeans |
|:-------------|--------:|-------:|-------:|---------:|----------:|
| A            |  0.1452 | 0.0305 | 0.4474 |   **0.0286** |    0.0518 |
| B            |  0.0850 | 0.0520 | 0.6044 |   0.0512 |    **0.0373** |
| C            |  **0.0147** | 0.0462 | 0.2375 |   0.0665 |    0.0849 |
| D            |  **0.0342** | 0.2169 | 0.0500 |   0.2135 |    0.2594 |
| E            |  0.2314 | 0.0371 | 0.0591 |   0.0615 |    **0.0353** |
| F            |  0.3162 | 0.0202 | **0.0002** |   **0.0002** |    0.0459 |
| G            |  0.7418 | **0.5524** | 0.5896 |   0.5568 |    0.6145 |

### SAD Statistics for Cuprite Dataset. 

| Statistics   |   AEsmm |    VCA |    PPI |   NFINDR |   KPmeans |
|:-------------|--------:|-------:|-------:|---------:|----------:|
| _Mean_       |  0.4503 | 0.1692 | 0.2840 |   **0.1623** |    0.1787 |
| _Std_        |  0.2932 | 0.0466 | **0.0000** |   0.0294 |    0.0406 |
| _Time_       | 20.7522 | 1.2126 | **0.9864** |   1.5111 |    5.8380 |
| _1-Ssim_     |  0.5799 | 0.7193 | 0.6668 |   **0.3919** |    0.7232 |
| _Mse_        |  0.0001 | 0.0019 | 0.0037 |   0.0018 |    **0.0000** |

### Comparison between the ground-truth and extracted endmembers using PPI, N-FINDR, VCA, using AAD for the simulate Dataset.

| Abundance   |   AEsmm |    VCA |    PPI |   NFINDR |   KPmeans |
|:------------|--------:|-------:|-------:|---------:|----------:|
| A           |  0.3561 | 0.3550 | 1.5042 |   **0.3057** |    0.3413 |
| B           |  1.3680 | **0.1797** | 0.4813 |   0.2411 |    0.1993 |
| C           |  0.6179 | 0.7754 | 0.7167 |   **0.4732** |    0.5074 |
| D           |  **0.4472** | 0.7061 | 0.9957 |   0.5814 |    0.5991 |
| E           |  0.4490 | 0.3918 | 0.5987 |   0.3905 |    **0.3381** |
| F           |  0.5508 | 0.2466 | 1.5464 |   0.2885 |    **0.2239** |
| G           |  **0.7211** | 1.4265 | 1.5657 |   1.4805 |    1.4541 |

### AAD Statistics for Cuprite Dataset. 

| Statistics   |   AEsmm |    VCA |    PPI |   NFINDR |   KPmeans |
|:-------------|--------:|-------:|-------:|---------:|----------:|
| _Mean_       |  0.7479 | 0.6684 | 1.0584 |   **0.5966** |    0.6089 |
| _Std_        |  0.1705 | 0.1163 | **0.0000** |   0.0877 |    0.1316 |
| _Time_       | 20.7522 | 1.2126 | **0.9864** |   1.5111 |    5.8380 |
| _1-Ssim_     |  0.6096 | 0.4220 | 0.6739 |   0.3919 |    **0.3799** |
| _Mse_        |  0.0001 | 0.0019 | 0.0037 |   0.0018 |    **0.0000** |

### Comparison between the ground-truth Laboratory Reflectances and extracted endmembers using PPI, N-FINDR, VCA using SID for the SaintAndre Dataset.

| Endmembers   |   AEsmm |    VCA |    PPI |   NFINDR |   KPmeans |
|:-------------|--------:|-------:|-------:|---------:|----------:|
| A            |  0.0198 | 0.0166 | 0.3274 |   **0.0036** |    0.0066 |
| B            |  0.0179 | 0.0477 | 0.4863 |   0.0354 |    **0.0058** |
| C            |  **0.0009** | 0.0069 | 0.1431 |   0.0014 |    0.0157 |
| D            |  **0.0045** | 0.0976 | 0.0065 |   0.0529 |    0.1718 |
| E            |  0.3667 | 0.0051 | 0.0049 |   0.0041 |    **0.0037** |
| F            |  0.0715 | 0.0006 | **0.0000** |   0.0008 |    0.0601 |
| G            |  **0.2609** | 0.4576 | 0.5214 |   0.4545 |    0.4970 |

### SID Statistics for Simulated Data. 

| Statistics   |   AEsmm |    VCA |    PPI |   NFINDR |   KPmeans |
|:-------------|--------:|-------:|-------:|---------:|----------:|
| _Mean_       |  4.3537 | 0.1503 | 0.2128 |   **0.1296** |    0.1926 |
| _Std_        |  8.2737 | 0.0437 | **0.0000** |   0.0427 |    0.0622 |
| _Time_       | 20.7522 | 1.2126 | **0.9864** |   1.5111 |    5.8380 |
| _1-Ssim_     |  0.7470 | 0.8316 | 0.6659 |   **0.5687** |    0.8112 |
| _Mse_        |  0.0001 | 0.0025 | 0.0037 |   0.0025 |    **0.0000** |

### Comparison between the ground-truth Laboratory Reflectances and extracted endmembers using PPI, N-FINDR, VCA using AID for the SaintAndre Dataset.

| Abundance   |   AEsmm |     VCA |     PPI |   NFINDR |   KPmeans |
|:------------|--------:|--------:|--------:|---------:|----------:|
| A           |  4.1985 |  5.2884 | 39.4878 |   **3.8757** |    4.1693 |
| B           |  6.6274 |  6.1070 |  6.1673 |   **3.6617** |    4.0375 |
| C           |  8.0631 |  **3.0707** |  6.5394 |   3.9149 |    6.5055 |
| D           |  1.9026 |  **1.6626** | 17.0055 |   4.1381 |    4.7037 |
| E           |  2.7238 |  3.9871 | **10.1399** |   3.4405 |    2.0321 |
| F           | **23.6248** |  8.6241 | 50.3475 |   4.7454 |    5.3246 |
| G           | **14.6100** | 47.0708 | 59.9497 |  46.6383 |   45.0140 |

### AID Statistics for Simulated Data. 

|    | Statistics   |   AEsmm |     VCA |     PPI |   NFINDR |   KPmeans |
|---:|:-------------|--------:|--------:|--------:|---------:|----------:|
|  0 | _Mean_       | 11.4047 | 12.9277 | 27.0910 |  **10.6937** |   11.8850 |
|  1 | _Std_        |  3.2732 |  5.7302 |  **0.0000** |   1.3466 |    2.9360 |
|  2 | _Time_       | 20.7522 |  1.2126 |  **0.9864** |   1.5111 |    5.8380 |
|  3 | _1-Ssim_     |  0.6425 |  0.4208 |  0.6631 |   0.3919 |    **0.3799** |
|  4 | _Mse_        |  0.0001 |  0.0016 |  0.0037 |   0.0018 |    **0.0000** |

![alt text](/mnt/Data/SU/data_results/SaintAndre/results/IMG/A_Endmember.png)

![alt text](/mnt/Data/SU/data_results/SaintAndre/results/IMG/A_Abundance.png)

![alt text](/mnt/Data/SU/data_results/SaintAndre/results/IMG/B_Endmember.png)

![alt text](/mnt/Data/SU/data_results/SaintAndre/results/IMG/B_Abundance.png)

![alt text](/mnt/Data/SU/data_results/SaintAndre/results/IMG/C_Endmember.png)

![alt text](/mnt/Data/SU/data_results/SaintAndre/results/IMG/C_Abundance.png)

![alt text](/mnt/Data/SU/data_results/SaintAndre/results/IMG/D_Endmember.png)

![alt text](/mnt/Data/SU/data_results/SaintAndre/results/IMG/D_Abundance.png)

![alt text](/mnt/Data/SU/data_results/SaintAndre/results/IMG/E_Endmember.png)

![alt text](/mnt/Data/SU/data_results/SaintAndre/results/IMG/E_Abundance.png)

![alt text](/mnt/Data/SU/data_results/SaintAndre/results/IMG/F_Endmember.png)

![alt text](/mnt/Data/SU/data_results/SaintAndre/results/IMG/F_Abundance.png)

![alt text](/mnt/Data/SU/data_results/SaintAndre/results/IMG/G_Endmember.png)

![alt text](/mnt/Data/SU/data_results/SaintAndre/results/IMG/G_Abundance.png)

