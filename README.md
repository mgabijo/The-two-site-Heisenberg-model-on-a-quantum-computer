<picture>
    <img src=".media/logo.png" width="200">
</picture>
  
# The two-site Heisenberg model studied using a quantum computer: A pedagogical introduction

Authors:
M G J Oliveira (1,2), T V C Antão (2,3), N M R Peres (1,4,5,6)

**1** - University of Minho, Physics Department and Center of Physics (CF-UM-UP), Campus of Gualtar, 3710-057 Braga, Portuga

**2** - LIP - Laboratório de Instrumentação e Física Experimental de Partículas, Escola de Ciências, Campus de Gualtar, Universidade do Minho, 
4701-057 Braga, Portugal

**3** - Department of Applied Physics, Aalto University, 02150 Espoo, Finland

**4** - International Iberian Nanotechnology Laboratory (INL), Av Mestre José Veiga, 4715-330 Braga, Portugal

**5** - POLIMA - Center for Polariton-driven Light-Matter Interactions, University of Southern Denmark, Campusvej 55, DK-5230 Odense M, Denmark

**6** - Danish Institute for Advanced Study, University of Southern Denmark, Campusvej 55, DK-5230 Odense M, Denmark


## Abstract

The two-site Heisenberg model has an extraordinarily simple analytical solution and is traditionally used as a benchmark against numerical methods, such as exact diagonalization and Monte Carlo methods.  In the same spirit, we benchmark three quantum algorithms that are implemented in a quantum computer against the analytical solution of this model. In particular, this presentation includes a description of the standard and iterative quantum phase estimation algorithms, as well as the variational quantum eigensolver. These quantum algorithms are introduced in a pedagogical fashion allowing newcomers to the subject, familiar with only the most basic quantum mechanical systems, to easily reproduce the presented results and apply the methods to other problems, thus building a seemingly under-appreciated path towards useful quantum algorithms through the lens of simulating and computing properties of physical quantum systems.

## File structure
This repository contains the code behind the article "The two-site Heisenberg model studied using a quantum computer: A pedagogical introduction" and its structure is the following:

- `qpea.py` - implements the quantum phase estimation algorithm for the two-site Heisenberg model;
- `ipea.py` - implements the (quantum) iterative phase estimation algorithm for the two-site Heisenberg model;
- `vqe.py` - implements the variational quantum eigensolver algorithm for the two-site Heisenberg model;
- `key.py` - should contain your IBM quantum experience token;
- `plots.ipynb` - contains the code to produce the plots.