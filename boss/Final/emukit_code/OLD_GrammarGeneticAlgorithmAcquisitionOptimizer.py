# define GA acquisition function optimizer!
import logging
from typing import Sequence, List, Tuple, Optional
import numpy as np
from emukit.core.optimization.acquisition_optimizer import AcquisitionOptimizerBase
from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.initial_designs import RandomDesign
from .cfg_parameter import CFGParameter

_log = logging.getLogger(__name__)

#TODO
# standardized fitness scores?
# parsimony_coefficient to stop getting massive trees
# allow hoist mutations to control bloat?
# allow point mutations?

class GrammarGeneticProgrammingOptimizer(AcquisitionOptimizerBase):
    """
    Optimizes the acquisition function using Genetic programming
    For use over grammar parameter spaces
    """
    def __init__(self, space: ParameterSpace, num_evolutions: int = 10, 
                 population_size: int = 5, tournament_prop: float = 0.5,
                 p_crossover: float = 0.9, p_subtree_mutation: float = 0.01
                ) -> None:
        """
        :param space: The parameter space spanning the search problem (has to consist of a single CFGParameter).
        :param num_steps: Maximum number of evolutions.
        :param num_init_points: Population size.
        :param tournament_prop: proportion of population randomly chosen from which to choose a tree to evolve
                                (larger gives faster convergence but smaller gives better diversity in the population)
        :p_crossover: probability of crossover evolution
        :p_subtree_mutation: probability of randomly mutating a subtree
        
        """
        super().__init__(space)
        #check that if parameter space is a single cfg param
        if len(space.parameters)!=1 or not isinstance(space.parameters[0],CFGParameter):
            raise ValueError("Genetic programming optimizer only for spaces consisting of a single cfg parameter")
        self.grammar = space.parameters[0].grammar
        # check probs for the evolutions sum to less than one (remaing prob is used to keep some trees unaltererd)
        if p_subtree_mutation + p_crossover >1:
            raise ValueError("Genetic operations sum to greater than one")
        self.p_subtree_mutation = p_subtree_mutation
        self.p_crossover = p_crossover
        self.p_reproduction = 1 - p_subtree_mutation - p_crossover
        # store as probs list for sampling
        self.probs = [self.p_subtree_mutation,self.p_crossover,self.p_reproduction]
            
        self.num_evolutions = num_evolutions
        self.population_size = population_size
        self.tournament_prop = tournament_prop
        
        
    def _optimize(self, acquisition: Acquisition) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementation of abstract method.
        See AcquisitionOptimizerBase._optimizer for parameter descriptions.
        See class docstring for implementation details.
        """
        # initialize population of tree

        random_design = RandomDesign(self.space)
        population = random_design.get_samples(self.population_size)
        # clac fitness for current population
        fitness_pop = acquisition.evaluate(unparse(population))
        standardized_fitness_pop = fitness_pop / sum(fitness_pop)
        # initialize best location and score so far
        X_max = population[np.argmax(fitness_pop)].reshape(-1,1) 
        acq_max = np.max(fitness_pop).reshape(-1,1) 
        _log.info("Starting local optimization of acquisition function {}".format(type(acquisition)))
        for step in range(self.num_evolutions):
            _log.info("Performing evolution step {}".format(step))
            # evolve populations
            population = self._evolve(population,standardized_fitness_pop)
            # recalc fitness
            fitness_pop = acquisition.evaluate(unparse(population))
            standardized_fitness_pop = fitness_pop / sum(fitness_pop)
            # update best location and score (if found better solution)
            acq_pop_max = np.max(fitness_pop)
            _log.info("best acqusition score in the new population".format(acq_pop_max))
            if acq_pop_max > acq_max[0][0]:
                acq_max[0][0] = acq_pop_max
                X_max[0] = X_pop[np.argmax(fitness_pop)]
                
        # return best solution from the whole optimization
        return X_max, acq_max
    
    def _evolve(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """ Performs a single evolution of the population, returnign the new population
        :param population: current population of trees (2d-array of strings)
        :param fitness:  fitness values of current points (2d-array)
        :return: a new population of trees (2d-array)
        """
        # perform genetic operations
        new_tree_pop = np.empty((len(population),1),dtype='<U53')
        i=0
        while i<len(population): # CAN WE PARALLELIZE THIS SOMEHOW
            # get the parent (stored as a string)
            parent=population[self._tournament(population,fitness)][0]
            # sample to see which operation to use
            r = np.random.choice(3, 1, p=self.probs)
            if r ==0:
                # perform subtree mutation
                new_tree_pop[i][0]=self._subtree_mutation(parent)
                i+=1
            elif r==1:
                # perform crossover
                child_1,child_2 = self._crossover(parent,donor)
                # update two memembers of the population (if enough left)
                new_tree_pop[i][0] = child_1
                if i<len(population)-2:
                    new_tree_pop[i+1][0] = child_2
                i+=2
            else:
                # perform reproduction, use tree unmodified
                new_tree_pop[i][0]=parent
                i+=1
        return new_tree_pop
    
    def _crossover(self, parent: str, population: np.ndarray) -> Tuple[str,str]:
        # randomly choose a subtree from the parent and replace
        # with a randomly chosen subtree from the donor
        # choose subtree to delete from paret
        subtree_node, subtree_index = rand_subtree(parent.self.grammar)
        # chop out subtree
        pre, sub, post = remove_subtree(parent,subtree_index)
        print(parent)
        print(pre)
        print(sub)
        print(post)
        # sample subtree from donor
        donor_subtree_index = rand_subtree_fixed_head(donor,subtree_node)
        # if no subtrees with right head node return False
        if not donor_subtree_index:
            return False, False
        else:
            donor_pre, donor_sub, donor_post = remove_subtree(donor,donor_subtree_index)
            # return the two new tree
            child_1 = pre + donor_sub + post 
            child_2 = donor_pre + sub + donor_post
            return child_1, child_2
      
    
    
    def _subtree_mutation(self, parent: str) -> str:
        # randomly choose a subtree from the parent and replace
        # with a new randomly generated subtree
        #
        # choose subtree to delete
        subtree_node, subtree_index = rand_subtree(parent,self.grammar)
        print(subtree_node)
        # chop out subtree
        pre,subtree,post = remove_subtree(parent,subtree_index)
        print(subtree)
        # generate new tree
        new = parse(self.space.parameters[0].sample_uniform(1))[0][0]
        # see if 
        new_subtree_index = rand_subtree_fixed_head(donor,subtree_node)

        new_subtree = self.space.parameters[0].sample_uniform(1,start=Nonterminal(subtree_node))
        print(new_subtree)
        new_subtree = parse(new_subtree,self.grammar)[0][0]
        print(new_subtree)
        print("\n")
        # return mutated tree
        return pre + new_subtree + post
         
   
    def _tournament(self, population:np.ndarray, fitness:np.ndarray) -> int :
        """ perfom a 'tournament' to select a suitiable parent from pop
        1) sample a sub-population of size tournament_size
        2) return index (in full pop) of the winner 
        """
        # size of tournament
        size = int(self.population_size * self.tournament_prop)
        # sample indicies
        contender_indicies = np.random.randint(self.population_size,size=size)
        contender_fitness = fitness[contender_indicies]
        # get best from this tournament and return their index
        return contender_indicies[np.argmax(contender_fitness)]
    
      
        
        
        
        
# helper function to swap between parse trees and strings
# e.g '2 + 1' <-> '(S (S (T 2)) (ADD +) (T 1))'
def parse(x,grammar):
    # grammar rep to tree rep
    string = x.split(' ')
    output=[]
    rd=ShiftReduceParser(grammar,0)
    for t in rd.parse(string):
        output.append(str(t))
    #get rid of annoying new lines and spaces 
    output = output[0].replace("\n","") 
    output = ' '.join(output.split())
    return output
parse = np.vectorize(parse)        
def unparse(tree):
    string=[]
    temp=""
    # perform single pass of tree
    for char in tree:
        if char==" ":
            temp=""
        elif char==")":
            if temp[-1]!= ")":
                string.append(temp)
            temp+=char
        else:
            temp+=char
    return " ".join(string)
unparse = np.vectorize(unparse) 






#helper function to choose a random subtree in a given tree
# returning the parent node of the subtree and its index
def rand_subtree(tree,grammar) -> int:
    # single pass through tree (stored as string) to look for the location of swappable_non_terminmals
    split_tree = tree.split(" ")
    swappable_indicies=[i for i in range(0,len(split_tree)) if split_tree[i][1:] in grammar.swappable_nonterminals]
    # randomly choose one of these non-terminals to replace its subtree
    r = np.random.randint(1,len(swappable_indicies))
    chosen_non_terminal = split_tree[swappable_indicies[r]][1:]
    chosen_non_terminal_index = swappable_indicies[r]
    # return chosen node and its index
    return chosen_non_terminal, chosen_non_terminal_index

# helper function to choose a random subtree from a given tree with a specific head node
# if no such subtree then return False, otherwise return the index of the subtree
def rand_subtree_fixed_head(tree, head_node) -> int:
    # single pass through tree (stored as string) to look for the location of swappable_non_terminmals
    split_tree = tree.split(" ")
    swappable_indicies=[i for i in range(0,len(split_tree)) if split_tree[i][1:]==head_node]
    if len(swappable_indicies)==0:
        # no such subtree
        return False
    else:
        # randomly choose one of these non-terminals 
        r = np.random.randint(1,len(swappable_indicies)) if len(swappable_indicies)>1 else 1
        chosen_non_terminal_index = swappable_indicies[r]
        return chosen_non_terminal_index

# helper function to remove a subtree from a tree (given its index)
# returning the str before and after the subtree
# i.e '(S (S (T 2)) (ADD +) (T 1))'
# becomes '(S (S (T 2)) ', '(T 1))'  after removing (ADD +)
def remove_subtree(tree,index)  -> Tuple[str,str,str]:
    split_tree = tree.split(" ")
    pre_subtree = " ".join(split_tree[:index])+" "
    #  get chars to the right of split
    right = " ".join(split_tree[index+1:])
    # remove chosen subtree
    # single pass to find the bracket matching the start of the split
    counter,current_index=1,0
    for char in right:
        if char=="(":
            counter+=1
        elif char==")":
            counter-=1
        if counter==0:
            break
        current_index+=1
    # retrun string after remover tree
    post_subtree = right[current_index+1:]
    # get removed tree
    removed = "".join(split_tree[index]) +" "+right[:current_index+1]
    return (pre_subtree, removed, post_subtree)
    