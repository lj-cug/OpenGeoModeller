# Titan2D

滑坡动力过程的MPI并行模拟，求解浅水方程，滑坡需要考虑源项的流变，主要模型有：

1. Mohr-Coulomb (MC)

2. Pouliquen-Forterre (PF)

3. Voellmy-Salm (VS)

## Competitive Advantages
```
Parallel/hybrid MPI/OpenMP execution; 
adaptive mesh refinement; 
integration with GIS interfaces such as GRASS; 
multiple rheology models (Mohr-Coulomb, Two-fluid Pitman-Le, Pouliquen-Forterre, Voellmy-Salm); 
free to use.
```

## references
```
Abani K. Patra,et al. 2018. Analyzing Complex Models Using Data and Statistics. https://doi.org/10.1007/978-3-319-93701-4_57

Abani Patra, et al. 2020. Comparative Analysis of the Structures and Outcomes of Geophysical Flow Models and Modeling Assumptions Using Uncertainty Quantification. Front. Earth Sci. 8: 275. doi: 10.3389/feart.2020.00275

Nikolay A. Simakov, et al. 2019. Modernizing Titan2D, a Parallel AMR Geophysical Flow Code to Support Multiple Rheologies and Extendability. https://doi.org/10.1007/978-3-030-34356-9_10

Bevilacqua A., Patra A. K., Bursik M. I., et al. 2019. Probabilistic forecasting of plausible debris flows from Nevado
de colima (Mexico) using data from the atenquique debris flow, 1955. Nat. Hazards Earth Syst. Sci. 19: 791–820.

```