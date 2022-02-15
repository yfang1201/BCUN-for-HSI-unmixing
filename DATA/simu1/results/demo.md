# Comparison of Unmixing method

## Envirionment Setup: 

Monte Carlo runs: 1 

Number of endmembers to estimate: 4 

Number of skewers (PPI): 1000 

Maximum number of iterations (N-FINDR): 12 

## SNR = : 20 

### Comparison between the ground-truth and extracted endmembers using PPI, N-FINDR, VCA using SAD for the simulate Dataset.

| Endmembers   |   BCUN |
|:-------------|-------:|
| A            | **0.1220** |
| B            | **0.0088** |
| C            | **0.0255** |
| D            | **0.0172** |

### SAD Statistics for Cuprite Dataset. 

| Statistics   |    BCUN |
|:-------------|--------:|
| _Mean_       |  **0.0434** |
| _Std_        |  **0.0000** |
| _Time_       | **16.7297** |
| _ssim_       |  **0.7653** |
| _Mse_        |  **0.0052** |

### Comparison between the ground-truth and extracted endmembers using PPI, N-FINDR, VCA, using AAD for the simulate Dataset.

| Abundance   |   BCUN |
|:------------|-------:|
| A           | **0.2154** |
| B           | **0.0958** |
| C           | **0.2580** |
| D           | **0.1777** |

### AAD Statistics for Cuprite Dataset. 

| Statistics   |    BCUN |
|:-------------|--------:|
| _Mean_       |  **0.1867** |
| _Std_        |  **0.0000** |
| _Time_       | **16.7297** |
| _ssim_       |  **0.7653** |
| _Mse_        |  **0.0052** |

### Comparison between the ground-truth Laboratory Reflectances and extracted endmembers using PPI, N-FINDR, VCA using SID for the Cuprite Dataset.

| Endmembers   |   BCUN |
|:-------------|-------:|
| A            | **0.0198** |
| B            | **0.0001** |
| C            | **0.0007** |
| D            | **0.0003** |

### SID Statistics for Simulated Data. 

| Statistics   |    BCUN |
|:-------------|--------:|
| _Mean_       |  **0.0052** |
| _Std_        |  **0.0000** |
| _Time_       | **16.7297** |
| _ssim_       |  **0.7653** |
| _Mse_        |  **0.0052** |

### Comparison between the ground-truth Laboratory Reflectances and extracted endmembers using PPI, N-FINDR, VCA using AID for the Cuprite Dataset.

| Abundance   |   BCUN |
|:------------|-------:|
| A           | **1.6717** |
| B           | **0.4323** |
| C           | **1.9155** |
| D           | **1.0037** |

### AID Statistics for Simulated Data. 

| Statistics   |    BCUN |
|:-------------|--------:|
| _Mean_       |  **1.2558** |
| _Std_        |  **0.0000** |
| _Time_       | **16.7297** |
| _ssim_       |  **0.7653** |
| _Mse_        |  **0.0052** |

![alt text](DATA/simu1/results/IMG/SNR=20_A_Endmember.png)

![alt text](DATA/simu1/results/IMG/SNR=20_A_Abundance.png)

![alt text](DATA/simu1/results/IMG/SNR=20_B_Endmember.png)

![alt text](DATA/simu1/results/IMG/SNR=20_B_Abundance.png)

![alt text](DATA/simu1/results/IMG/SNR=20_C_Endmember.png)

![alt text](DATA/simu1/results/IMG/SNR=20_C_Abundance.png)

![alt text](DATA/simu1/results/IMG/SNR=20_D_Endmember.png)

![alt text](DATA/simu1/results/IMG/SNR=20_D_Abundance.png)

