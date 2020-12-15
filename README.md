# Comparison of Unmixing method

**Envirionment Setup:**

Monte Carlo runs: 3 

Number of endmembers to estimate: 4 

Number of skewers (PPI): 10 

Maximum number of iterations (N-FINDR): 3 

### Parameters used in each GAEE versions

| Parameters            |   GAEE |   GAEE-IVFm |   GAEE-VCA |   GAEE-IVFm-VCA |
|:----------------------|-------:|------------:|-----------:|----------------:|
| Population Size       | 100    |       100   |      100   |           100   |
| Number of Generations |   5    |         5   |        5   |             5   |
| Crossover Probability |   0.7  |         0.7 |        0.5 |             0.5 |
| Mutation Probability  |   0.05 |         0.1 |        0.3 |             0.1 |

![alt text](./IMG/Convergence.png)

### Comparison between the ground-truth and extracted endmembers using PPI, N-FINDR, VCA, GAEE, GAEE-IVFm using SAD for the simulate Dataset.

| Endmembers   |    PPI |   NFINDR |    VCA |   GAEE |   GAEE-IVFm |   GAEE-VCA |   GAEE-IVFm-VCA |
|:-------------|-------:|---------:|-------:|-------:|------------:|-----------:|----------------:|
| A            | 0.1660 |   0.0259 | 0.0223 | 0.0089 |      **0.0077** |     0.0223 |          0.0223 |
| B            | 0.1417 |   **0.0447** | 0.0686 | 0.0531 |      0.0496 |     0.0686 |          0.0686 |
| C            | 0.0497 |   0.0260 | **0.0246** | 0.0305 |      0.0615 |     **0.0246** |          **0.0246** |
| D            | 0.0928 |   0.0232 | **0.0167** | 0.0348 |      0.0357 |     **0.0167** |          **0.0167** |

### SAD Statistics for Cuprite Dataset. 

| Statistics   |      PPI |   NFINDR |     VCA |    GAEE |   GAEE-IVFm |   GAEE-VCA |   GAEE-IVFm-VCA |
|:-------------|---------:|---------:|--------:|--------:|------------:|-----------:|----------------:|
| _Mean_       |   0.1126 | nan      |  0.0394 |  0.0606 |      0.0670 |     **0.0331** |          **0.0331** |
| _Std_        |   **0.0000** | nan      |  0.0091 |  0.0268 |      0.0267 |     **0.0000** |          **0.0000** |
| _p-value_    | -23.2168 | nan      |  0.0000 | **-1.3938** |     -1.5447 |     1.9919 |          1.9919 |
| Gain         |  70.6131 | nan      | 15.9571 | 45.4141 |     50.6406 |     **0.0000** |          **0.0000** |
| _Time_       |   3.1736 |   **3.1626** |  3.2041 |  3.3139 |      3.3360 |     3.3004 |          3.3273 |

### Comparison between the ground-truth and extracted endmembers using PPI, N-FINDR, VCA, GAEE, GAEE-IVFm using AAD for the simulate Dataset.

| Abundance   |    PPI |   NFINDR |    VCA |   GAEE |   GAEE-IVFm |   GAEE-VCA |   GAEE-IVFm-VCA |
|:------------|-------:|---------:|-------:|-------:|------------:|-----------:|----------------:|
| A           | 0.7897 |   0.7951 | **0.7619** | 0.7633 |      0.8143 |     **0.7619** |          **0.7619** |
| B           | 0.6571 |   **0.6156** | 0.6613 | 0.7236 |      0.6267 |     0.6613 |          0.6613 |
| C           | 0.7499 |   0.5998 | 0.5938 | **0.5818** |      0.6541 |     0.5938 |          0.5938 |
| D           | 0.9774 |   **0.6194** | 0.6222 | 0.6217 |      0.6310 |     0.6222 |          0.6222 |

### AAD Statistics for Cuprite Dataset. 

| Statistics   |      PPI |   NFINDR |    VCA |    GAEE |   GAEE-IVFm |   GAEE-VCA |   GAEE-IVFm-VCA |
|:-------------|---------:|---------:|-------:|--------:|------------:|-----------:|----------------:|
| _Mean_       |   0.7935 | nan      | 0.6690 |  0.6868 |      0.6992 |     **0.6598** |          **0.6598** |
| _Std_        |   **0.0000** | nan      | 0.0184 |  0.0521 |      0.0638 |     **0.0000** |          **0.0000** |
| _p-value_    | -24.1913 | nan      | 0.0000 | **-1.8252** |     -2.4771 |     1.7836 |          1.7836 |
| Gain         |  16.8519 | nan      | 1.3726 |  3.9349 |      5.6320 |     **0.0000** |          **0.0000** |
| _Time_       |   3.1736 |   **3.1626** | 3.2041 |  3.3139 |      3.3360 |     3.3004 |          3.3273 |

### Comparison between the ground-truth Laboratory Reflectances and extracted endmembers using PPI, N-FINDR, VCA, GAEE, GAEE-IVFm using SID for the Cuprite Dataset.

| Endmembers   |    PPI |   NFINDR |    VCA |   GAEE |   GAEE-IVFm |   GAEE-VCA |   GAEE-IVFm-VCA |
|:-------------|-------:|---------:|-------:|-------:|------------:|-----------:|----------------:|
| A            | 0.0410 |   0.0010 | 0.0007 | **0.0001** |      **0.0001** |     0.0007 |          0.0007 |
| B            | 0.0319 |   **0.0035** | 0.0078 | 0.0046 |      0.0045 |     0.0078 |          0.0078 |
| C            | 0.0025 |   0.0007 | **0.0006** | 0.0010 |      0.0048 |     **0.0006** |          **0.0006** |
| D            | 0.0088 |   0.0005 | **0.0003** | 0.0012 |      0.0013 |     **0.0003** |          **0.0003** |

### SID Statistics for Simulated Data. 

| Statistics   |      PPI |   NFINDR |     VCA |    GAEE |   GAEE-IVFm |   GAEE-VCA |   GAEE-IVFm-VCA |
|:-------------|---------:|---------:|--------:|--------:|------------:|-----------:|----------------:|
| _Mean_       |   0.0210 |   0.0029 |  0.0032 |  0.0063 |      0.0071 |     **0.0024** |          **0.0024** |
| _Std_        |   **0.0000** |   0.0033 |  0.0010 |  0.0049 |      0.0046 |     **0.0000** |          **0.0000** |
| _p-value_    | -41.2028 |   0.2297 |  0.0000 | **-1.2874** |     -1.3047 |     1.9996 |          1.9996 |
| Gain         |  70.6131 | nan      | 15.9571 | 45.4141 |     50.6406 |     **0.0000** |          **0.0000** |
| _Time_       |   3.1736 |   **3.1626** |  3.2041 |  3.3139 |      3.3360 |     3.3004 |          3.3273 |

### Comparison between the ground-truth Laboratory Reflectances and extracted endmembers using PPI, N-FINDR, VCA, GAEE, GAEE-IVFm using AID for the Cuprite Dataset.

| Abundance   |     PPI |   NFINDR |    VCA |   GAEE |   GAEE-IVFm |   GAEE-VCA |   GAEE-IVFm-VCA |
|:------------|--------:|---------:|-------:|-------:|------------:|-----------:|----------------:|
| A           | 10.2861 |   **0.0000** | 9.5105 | 8.9252 |      8.0548 |     9.9115 |          9.9115 |
| B           |  5.4329 |   **0.0000** | 5.7467 | 5.1741 |      4.8431 |     4.5110 |          4.5110 |
| C           |  6.7058 |   **0.0000** | 4.7315 | 4.9698 |      3.7661 |     4.6500 |          4.6500 |
| D           |  9.8918 |   **4.0759** | 4.9373 | 4.7669 |      8.0910 |     7.1416 |          7.1416 |

### AID Statistics for Simulated Data. 

| Statistics   |      PPI |   NFINDR |     VCA |   GAEE |   GAEE-IVFm |   GAEE-VCA |   GAEE-IVFm-VCA |
|:-------------|---------:|---------:|--------:|-------:|------------:|-----------:|----------------:|
| _Mean_       |   8.0792 |   **4.3323** |  6.4145 | 6.2937 |      6.5865 |     6.5535 |          6.5535 |
| _Std_        |   **0.0000** |   2.3429 |  0.5527 | 0.6996 |      0.9140 |     **0.0000** |          **0.0000** |
| _p-value_    | -17.4264 |   1.2548 |  0.0000 | 0.4499 |     **-0.5536** |    -1.4555 |         -1.4555 |
| Gain         |  13.4461 | nan      | **-2.6673** | 0.0000 |      1.7667 |    -4.0960 |         -4.0960 |
| _Time_       |   3.1736 |   **3.1626** |  3.2041 | 3.3139 |      3.3360 |     3.3004 |          3.3273 |

![alt text](./IMG/A_Endmember.png)

![alt text](./IMG/B_Endmember.png)

![alt text](./IMG/C_Endmember.png)

![alt text](./IMG/D_Endmember.png)

