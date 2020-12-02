import random
import numpy as np
class GeneticEngine():
  '''
    Parameters:
    - Population size
    - Gene size
    - Method to generate new individual
    - Specify the selection method to be used
    - Specify the reproduction method to be used
    - Method to mutate an individual
    - Chance to mutate an individual
    - Function of fitness to be used
    - Method to recognize when end criteria is met
  '''

  def __init__(
      self,
      population_size,
      gene_size,
      f_nw_indv,
      f_sel,
      f_repr,
      f_mut,
      mut_prob,
      f_fit,
      f_end_criteria 
    ):
    self.population_size = population_size
    self.gene_size = gene_size
    self.f_nw_indv = f_nw_indv
    self.f_sel = self.tournament_selection if f_sel == "tournament" else self.roulette_selection
    self.f_repr = self.crossover if f_repr == "crossover" else self.variant_crossover
    self.f_mut = f_mut
    self.mut_prob = mut_prob
    self.f_fit = f_fit
    self.f_end_criteria = f_end_criteria

    # Generate initial population
    self.init_population()

    # Initialize metrics
    self.historic_min_metric = []
    self.historic_mean_metric = []
    self.historic_max_metric = []

    # Initialize best result found
    self.best = None

  def init_population(self):
    self.population = [ self.f_nw_indv() for i in range(self.population_size) ]

  def gen_rand_with_memory(self, prev_idxs):
    while (True):
      new_rand = random.randint(0, self.population_size - 1)
      flag = True
      for idx in prev_idxs:
        if (new_rand == idx):
          flag = False
          break
      if (flag):
          return new_rand

  def roulette_selection(self):
    parent_idxs = []
    for i in range(0, 2):
        parent_idxs.append(self.gen_rand_with_memory(parent_idxs))
    return [self.population[idx] for idx in parent_idxs] 

  def tournament_selection(self):
    parent_idxs = []
    for i in range(0,2):
      prospect_idxs = []
      fit_vals = []
      for i in range(0,min(5,len(self.population)) - len(parent_idxs) ):
        prospect_idxs.append(self.gen_rand_with_memory(prospect_idxs + parent_idxs))
        fit_vals.append(self.f_fit(self.population[prospect_idxs[-1]]))
      parent_idxs.append(prospect_idxs[np.argmax(fit_vals)])
    return [self.population[idx] for idx in parent_idxs]

  def crossover(self, parents):
    r_val = random.randint(1, self.gene_size - 2)

    # returning a list with a single new element
    return [ parents[0][0:r_val] + parents[1][r_val:] ]
  
  def variant_crossover(self, parents):
    r_val = random.randint(1, self.gene_size - 2)

    # returning a list with two new elements
    return [ parents[0][0:r_val] + parents[1][r_val:], parents[1][0:r_val] + parents[0][r_val:] ]
  
  def eval_population(self):
    fitness = []
    for individual in self.population:
      fitness.append(self.f_fit(individual))
    
    idx_min = np.argmin(fitness)
    c_mean = np.mean(fitness)
    idx_max = np.argmax(fitness)

    self.historic_min_metric.append(fitness[idx_min])
    self.historic_mean_metric.append(c_mean)
    self.historic_max_metric.append(fitness[idx_max])

    # Assuming Maximization approach 
    if self.best is None or np.max(self.historic_max_metric) < fitness[idx_max]:
      self.best = self.population[idx_max]

    return [ (idx_min,fitness[idx_min]) , (-1,c_mean), (idx_max,fitness[idx_max]) ]
  
  def select_and_reproduction(self):
    new_population = []
    print("\t- Selection,Reproduction & Mutation")
    while (len(new_population) < self.population_size):
      new_individuals = self.f_repr(self.f_sel())
      for indv in new_individuals:
        if random.random() < self.mut_prob:
          new_population.append(self.f_mut(indv))
        else:
          new_population.append(indv)

    self.population = new_population
  
  def run(self, generations):
    self.historic_min_metric = []
    self.historic_mean_metric = []
    self.historic_max_metric = []
    self.best = None

    for g in range(0,generations):
      print(f"Generation: {g+1}")

      print("Evaluation")
      # Eval fitness of current population
      current_results = self.eval_population()

      # Verify if we found an acceptable individual
      if self.f_end_criteria(current_results[-1][1]):
        print("Criteria met")
        return self.population[current_results[-1][0]], current_results[-1][1]

      # Apply selection and reproduction methods
      self.select_and_reproduction()      
    
    print("Criteria not met returning the best found")
    return self.best, np.max(self.historic_max_metric)
