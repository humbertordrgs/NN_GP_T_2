# # Neuronal Network and Genetic Programming Task 2
This project with has the objective of showcasing the qualities of a genetic algorithm and it's versatility if implemented correctly.

## Project structure

### Environment
```
- conda
- python 3.8.3
- jupyter
- numpy 
- pytorch
- plotly
- plotly-orca (for static rendering of charts)
```

### Solution Approach
This solution consists of a main module named `GeneticEngine` destined to encapsulate the main logic needed to initialize, implement and run a genetic algorithm solution over as many different types of problems as possible (located in `utils.GeneticEngine`). We present 3 different problems to solve using the same engine:
1) `dec_to_bin`: a simple problem where given a decimal number it is necesary for the algorithm to find it's binary representation.
2) `find_the_word`: another simple problem where given a string the objective is for the algorithm to find said value.
3) `neural_network_optimizer`: a more complex problem, one of the objectives of our research project is to explore different ways to optimize the hyperparameters of a neural network model. In this work we attempt to show that these types of algorithms are viable as a solution for finding a very good set of hyperparameters.

### Analysis and Results
The detailed info of this experiments can be found on their respective files.
