import itertools
from typing import Iterable, Union, Tuple, List

import numpy as np

from emukit.core.parameter import Parameter


from nltk.parse import ShiftReduceParser

class CFG_AS_Parameter(Parameter):
    """
    A class for inputs consisting of a string of kerenls that can be generated from a context free grammar (CFG)
    This is a specific class to employ the samplign strategy of lu et al 2018 (i.e unformly and fixed length)
    """
    def __init__(self, name: str, grammar: object):
        """
        :param name: Name of parameter
        :param grammar: cfg inducing the set of possible strings

        These other parameters control how we sample from the grammar (see self.sample_uniform)
        :max_length: maximum length of sampled tree from grammar. length means number of terminals
        :min_length: minimum length of sampled tree from grammar
        :cfactor: smaller cfactor provides smaller sequences (on average)
        """
        self.name = name
        self.grammar = grammar


    @property
    def bounds(self) -> List[Tuple]:
        """
        MAYBE COULD RETURN THE ALPHABET?
        """
        raise NotImplemented

    def sample_uniform(self, point_count: int=1) -> np.ndarray:
        """
        Generates multiple (unqiue) random strings from the grammar of fixed length
        :returns: Generated points with shape (point_count, 1)
        """

        samples = np.ones((point_count,1),dtype=object)
        for i in range(0,point_count):
            kerns = ["k1","k2","k3","k4"]
            ops = ["+","*","BREAK"]
            sample=[]
            for j in range(0,5):
                # sample a kern
                sample.append(kerns[np.random.randint(0,4)])
                # sample an op
                new_op = ops[np.random.randint(0,3)]
                if new_op=="BREAK":
                    break
                else:
                    sample.append(new_op)
            # if ends on an op then needs final kern
            if sample[-1] in ["+","*"]:
                sample.append(kerns[np.random.randint(0,4)])
            # parse the sample
            output=[]
            rd=ShiftReduceParser(self.grammar,0)
            for t in rd.parse(sample):
                output.append(str(t))
            #get rid of new lines and spaces 
            output = output[0].replace("\n","") 
            output = ' '.join(output.split())
            samples[i][0]=output
        return samples
