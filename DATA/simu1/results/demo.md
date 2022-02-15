# Comparison of Unmixing method

## Envirionment Setup: 

Monte Carlo runs: 1 

Number of endmembers to estimate: 4 

Number of skewers (PPI): 1000 

Maximum number of iterations (N-FINDR): 12 

## SNR = : 10 

### Comparison between the ground-truth and extracted endmembers using PPI, N-FINDR, VCA using SAD for the simulate Dataset.

| Endmembers   |   BCUN |
|:-------------|-------:|
| A            | **0.1405** |
| B            | **0.2977** |
| C            | **0.1310** |
| D            | **0.2540** |

### SAD Statistics for Cuprite Dataset. 

| Statistics   |   BCUN |
|:-------------|-------:|
| _Mean_       | **0.2058** |
| _Std_        | **0.0000** |
| _Time_       | **0.8000** |
| _ssim_       | **0.2620** |
| _Mse_        | **0.0554** |

### Comparison between the ground-truth and extracted endmembers using PPI, N-FINDR, VCA, using AAD for the simulate Dataset.

| Abundance   |   BCUN |
|:------------|-------:|
| A           | **0.7127** |
| B           | **0.6140** |
| C           | **0.6823** |
| D           | **0.6877** |

### AAD Statistics for Cuprite Dataset. 

| Statistics   |   BCUN |
|:-------------|-------:|
| _Mean_       | **0.6742** |
| _Std_        | **0.0000** |
| _Time_       | **0.8000** |
| _ssim_       | **0.2620** |
| _Mse_        | **0.0554** |

### Comparison between the ground-truth Laboratory Reflectances and extracted endmembers using PPI, N-FINDR, VCA using SID for the Cuprite Dataset.

| Endmembers   |   BCUN |
|:-------------|-------:|
| A            | **0.0195** |
| B            | **0.1340** |
| C            | **0.0167** |
| D            | **0.0852** |

### SID Statistics for Simulated Data. 

| Statistics   |   BCUN |
|:-------------|-------:|
| _Mean_       | **0.0639** |
| _Std_        | **0.0000** |
| _Time_       | **0.8000** |
| _ssim_       | **0.2620** |
| _Mse_        | **0.0554** |

### Comparison between the ground-truth Laboratory Reflectances and extracted endmembers using PPI, N-FINDR, VCA using AID for the Cuprite Dataset.

| Abundance   |   BCUN |
|:------------|-------:|
| A           | **5.7811** |
| B           | **4.8604** |
| C           | **6.4084** |
| D           | **6.1259** |

### AID Statistics for Simulated Data. 

| Statistics   |   BCUN |
|:-------------|-------:|
| _Mean_       | **5.7940** |
| _Std_        | **0.0000** |
| _Time_       | **0.8000** |
| _ssim_       | **0.2620** |
| _Mse_        | **0.0554** |

![alt text](DATA/simu1/results/IMG/SNR=10_A_Endmember.png)

![alt text](DATA/simu1/results/IMG/SNR=10_A_Abundance.png)

![alt text](DATA/simu1/results/IMG/SNR=10_B_Endmember.png)

![alt text](DATA/simu1/results/IMG/SNR=10_B_Abundance.png)

![alt text](DATA/simu1/results/IMG/SNR=10_C_Endmember.png)

![alt text](DATA/simu1/results/IMG/SNR=10_C_Abundance.png)

![alt text](DATA/simu1/results/IMG/SNR=10_D_Endmember.png)

![alt text](DATA/simu1/results/IMG/SNR=10_D_Abundance.png)

