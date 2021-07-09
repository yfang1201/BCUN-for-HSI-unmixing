# Comparison of Unmixing method

## Envirionment Setup: 

Monte Carlo runs: 1 

Number of endmembers to estimate: 4 

Number of skewers (PPI): 1000 

Maximum number of iterations (N-FINDR): 12 

Maximum number of iterations (KP) : 50 

Maximum number of iterations (BCUN/BCUN0) : 60 

Maximum number of epoch (BCUN/BCUN0) : 500 

### Comparison between the ground-truth and extracted endmembers using PPI, N-FINDR, VCA using SAD for the simulate Dataset.

| Endmembers   |    VCA |   AEsmm |
|:-------------|-------:|--------:|
| A            | **0.1870** |  0.2852 |
| B            | **0.0422** |  2.7713 |
| C            | **0.0589** |  0.1136 |
| D            | 0.1865 |  **0.0682** |

### SAD Statistics for Cuprite Dataset. 

| Statistics   |     VCA |     AEsmm |
|:-------------|--------:|----------:|
| _Mean_       |  **0.1187** |    0.8096 |
| _Std_        |  **0.0000** |    **0.0000** |
| _Time_       | 28.2939 | **1857.5949** |
| _1-Ssim_     |  **0.4282** |    0.6504 |
| _Mse_        |  **0.0001** |    0.0002 |

### Comparison between the ground-truth and extracted endmembers using PPI, N-FINDR, VCA, using AAD for the simulate Dataset.

| Abundance   |    VCA |   AEsmm |
|:------------|-------:|--------:|
| A           | **0.5762** |  0.6592 |
| B           | **0.2851** |  0.9687 |
| C           | **0.4697** |  0.6032 |
| D           | **0.6238** |  0.7903 |

### AAD Statistics for Cuprite Dataset. 

| Statistics   |     VCA |     AEsmm |
|:-------------|--------:|----------:|
| _Mean_       |  **0.4887** |    0.7554 |
| _Std_        |  **0.0000** |    **0.0000** |
| _Time_       | 28.2939 | **1857.5949** |
| _1-Ssim_     |  **0.4282** |    0.6504 |
| _Mse_        |  **0.0001** |    0.0002 |

### Comparison between the ground-truth Laboratory Reflectances and extracted endmembers using PPI, N-FINDR, VCA using SID for the Cuprite Dataset.

| Endmembers   |    VCA |   AEsmm |
|:-------------|-------:|--------:|
| A            | **0.0444** |  0.1912 |
| B            | **0.0035** |  0.1433 |
| C            | **0.0563** |  0.0634 |
| D            | 0.0547 |  **0.0047** |

### SID Statistics for Simulated Data. 

| Statistics   |     VCA |     AEsmm |
|:-------------|--------:|----------:|
| _Mean_       |  **0.0398** |    0.1007 |
| _Std_        |  **0.0000** |    **0.0000** |
| _Time_       | 28.2939 | **1857.5949** |
| _1-Ssim_     |  **0.4282** |    0.6504 |
| _Mse_        |  **0.0001** |    0.0002 |

### Comparison between the ground-truth Laboratory Reflectances and extracted endmembers using PPI, N-FINDR, VCA using AID for the Cuprite Dataset.

| Abundance   |    VCA |   AEsmm |
|:------------|-------:|--------:|
| A           | **0.9716** | 10.7577 |
| B           | **1.0367** |  2.9114 |
| C           | **2.2262** |  3.4295 |
| D           | **2.2298** |  4.8652 |

### AID Statistics for Simulated Data. 

| Statistics   |     VCA |     AEsmm |
|:-------------|--------:|----------:|
| _Mean_       |  **1.6161** |    5.4910 |
| _Std_        |  **0.0000** |    **0.0000** |
| _Time_       | 28.2939 | **1857.5949** |
| _1-Ssim_     |  **0.4282** |    0.6252 |
| _Mse_        |  **0.0001** |    0.0002 |

![alt text](/mnt/Data/SU/data_results/Urban/results_4/IMG/A_Endmember.png)

![alt text](/mnt/Data/SU/data_results/Urban/results_4/IMG/A_Abundance.png)

![alt text](/mnt/Data/SU/data_results/Urban/results_4/IMG/B_Endmember.png)

![alt text](/mnt/Data/SU/data_results/Urban/results_4/IMG/B_Abundance.png)

![alt text](/mnt/Data/SU/data_results/Urban/results_4/IMG/C_Endmember.png)

![alt text](/mnt/Data/SU/data_results/Urban/results_4/IMG/C_Abundance.png)

![alt text](/mnt/Data/SU/data_results/Urban/results_4/IMG/D_Endmember.png)

![alt text](/mnt/Data/SU/data_results/Urban/results_4/IMG/D_Abundance.png)

