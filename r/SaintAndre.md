# Comparison of Unmixing method

## Envirionment Setup: 

Monte Carlo runs: 5 

Number of endmembers to estimate: 6 

Number of skewers (PPI): 1000 

Maximum number of iterations (N-FINDR): 18 

### Comparison between the ground-truth and extracted endmembers using PPI, N-FINDR, VCA using SAD for the simulate Dataset.

| Endmembers   |   AEsmm |    VCA |    PPI |   NFINDR |   KPmeans |
|:-------------|--------:|-------:|-------:|---------:|----------:|
| A            |  0.2481 | 0.0351 | 0.4474 |   **0.0286** |    0.0448 |
| B            |  0.0542 | 0.6791 | 0.6019 |   **0.0003** |    0.0402 |
| C            |  0.1415 | **0.0462** | 0.2375 |   0.1195 |    0.0883 |
| D            |  0.0815 | 0.1022 | **0.0500** |   0.2570 |    0.0564 |
| E            |  0.0677 | 0.0373 | 0.0591 |   0.0356 |    **0.0231** |
| F            |  0.3796 | 0.0322 | **0.0002** |   **0.0002** |    0.6496 |

### SAD Statistics for Cuprite Dataset. 

| Statistics   |   AEsmm |    VCA |    PPI |   NFINDR |   KPmeans |
|:-------------|--------:|-------:|-------:|---------:|----------:|
| _Mean_       |  0.2211 | 0.1796 | 0.2327 |   **0.1413** |    0.1841 |
| _Std_        |  0.1161 | 0.0294 | **0.0000** |   0.1027 |    0.0422 |
| _Time_       |  9.4208 | 1.1536 | **0.9520** |   1.3209 |    5.3723 |
| _1-Ssim_     |  0.7362 | 0.7735 | 0.7007 |   0.6282 |    **0.2534** |
| _Mse_        |  0.0003 | 0.0035 | 0.0037 |   0.0016 |    **0.0000** |

### Comparison between the ground-truth and extracted endmembers using PPI, N-FINDR, VCA, using AAD for the simulate Dataset.

| Abundance   |   AEsmm |    VCA |    PPI |   NFINDR |   KPmeans |
|:------------|--------:|-------:|-------:|---------:|----------:|
| A           |  0.8898 | 0.2915 | 1.5037 |   0.3211 |    **0.2396** |
| B           |  0.5962 | 0.4235 | 0.4760 |   **0.1778** |    0.4873 |
| C           |  0.5036 | 0.4564 | 0.7167 |   0.6685 |    **0.2697** |
| D           |  0.5641 | **0.3191** | 1.0059 |   0.8433 |    0.3495 |
| E           |  0.5090 | 0.4466 | 0.6090 |   0.4152 |    **0.2797** |
| F           |  1.4561 | 1.4928 | 1.5491 |   **0.2222** |    1.4088 |

### AAD Statistics for Cuprite Dataset. 

| Statistics   |   AEsmm |    VCA |    PPI |   NFINDR |   KPmeans |
|:-------------|--------:|-------:|-------:|---------:|----------:|
| _Mean_       |  0.7822 | 0.6678 | 0.9767 |   0.6277 |    **0.5716** |
| _Std_        |  0.0919 | 0.0960 | **0.0000** |   0.1458 |    0.0468 |
| _Time_       |  9.4208 | 1.1536 | **0.9520** |   1.3209 |    5.3723 |
| _1-Ssim_     |  0.5010 | 0.2939 | 0.7092 |   0.4630 |    **0.2534** |
| _Mse_        |  0.0004 | 0.0026 | 0.0037 |   0.0016 |    **0.0000** |

### Comparison between the ground-truth Laboratory Reflectances and extracted endmembers using PPI, N-FINDR, VCA using SID for the SaintAndre Dataset.

| Endmembers   |   AEsmm |    VCA |    PPI |   NFINDR |   KPmeans |
|:-------------|--------:|-------:|-------:|---------:|----------:|
| A            |  0.0763 | 0.0014 | 0.3274 |   **0.0008** |    0.0020 |
| B            |  0.0070 | 0.5011 | 0.4863 |   **0.0000** |    0.0044 |
| C            |  0.0864 | **0.0066** | 0.1431 |   0.0157 |    0.0500 |
| D            |  0.0280 | 0.0123 | 0.0065 |   0.0883 |    **0.0060** |
| E            |  0.0082 | 0.0021 | 0.0049 |   0.0018 |    **0.0011** |
| F            |  0.1965 | 0.0030 | **0.0000** |   **0.0000** |    0.5817 |

### SID Statistics for Simulated Data. 

| Statistics   |   AEsmm |    VCA |    PPI |   NFINDR |   KPmeans |
|:-------------|--------:|-------:|-------:|---------:|----------:|
| _Mean_       |  0.1594 | 0.1590 | 0.1614 |   **0.0804** |    0.1562 |
| _Std_        |  0.1611 | 0.0452 | **0.0000** |   0.0815 |    0.0507 |
| _Time_       |  9.4208 | 1.1536 | **0.9520** |   1.3209 |    5.3723 |
| _1-Ssim_     |  0.7362 | 0.7735 | 0.7007 |   0.6282 |    **0.2534** |
| _Mse_        |  0.0003 | 0.0035 | 0.0037 |   0.0016 |    **0.0000** |

### Comparison between the ground-truth Laboratory Reflectances and extracted endmembers using PPI, N-FINDR, VCA using AID for the SaintAndre Dataset.

| Abundance   |   AEsmm |     VCA |     PPI |   NFINDR |   KPmeans |
|:------------|--------:|--------:|--------:|---------:|----------:|
| A           |  8.3042 |  4.3487 | 37.2782 |   3.5814 |    **2.4661** |
| B           |  8.7874 |  3.7762 |  5.3415 |   **1.5837** |    7.0808 |
| C           |  5.7536 |  2.8988 |  6.5743 |   7.1997 |    **2.0156** |
| D           |  2.3472 |  2.7843 | 17.4889 |   7.7488 |    **1.9808** |
| E           |  9.2668 |  3.7856 | 10.3888 |   3.0392 |    **1.9185** |
| F           | **16.2074** | 40.8889 | 54.2279 |   4.5588 |   25.3300 |

### AID Statistics for Simulated Data. 

|    | Statistics   |   AEsmm |     VCA |     PPI |   NFINDR |   KPmeans |
|---:|:-------------|--------:|--------:|--------:|---------:|----------:|
|  0 | _Mean_       |  9.6139 | **13.0728** | 21.8833 |   7.8656 |    8.7437 |
|  1 | _Std_        |  1.4923 |  5.7708 |  **0.0000** |   2.5646 |    1.7742 |
|  2 | _Time_       |  9.4208 |  1.1536 |  **0.9520** |   1.3209 |    5.3723 |
|  3 | _1-Ssim_     |  0.5186 |  0.4678 |  0.7092 |   0.4630 |    **0.2534** |
|  4 | _Mse_        |  0.0003 |  0.0022 |  0.0037 |   0.0016 |    **0.0000** |

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

