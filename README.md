# Security Analysis of Safe & Seldonian Reinforcement Learning Algorithms

This code implements the experiments described in the paper
"[Security Analysis of Safe & Seldonian Reinforcement Learning Algorithms](https://people.cs.umass.edu/~pinar/ozisik.neurips.2020.pdf)"
by [A. Pinar Ozisik](https://cs.umass.edu/~pinar), and [Philip S. Thomas](https://people.cs.umass.edu/~pthomas/).

Requirements
----
- \>= Python 3.6 with pip
- R

Data Collection
----
This code collects data using a slightly modified version of Jinyu Xie's [Simglucose v0.2.1](https://github.com/jxx123/simglucose). Xie's code, found in the contents of folder SimGlucose, is incorporated into this repository for convenience.  

Setup
----
1. Download the requirements: ```pip install -r requirements.txt```

2. To replicate the results in the paper, from the root directory of this project, run:```python run.py 0```

3. To run the same experiment with a random behavior and evaluation policy, from the root directory of this project, run:```python run.py 1```

Results
----
The plot that will be generated by the code will be placed in "resuts/final_results.pdf"

Configs
----
constants.py specifies all the hyperparameters used in the experiment and can be changed.

TODO
----
Implemantation of softmax action selection to run experiments with a random behavior and evaluation policy