from gpflow.kernels import Kernel
from gpflow.utilities import positive
from gpflow import Parameter
import tensorflow as tf
import numpy as np
from tensorflow_probability import bijectors as tfb

class StringKernel(Kernel):
    """
    Code to run the SSK of Moss et al. 2020 with gpflow
    
   with hyperparameters:
    1) match_decay float
        decrease the contribution of long subsequences
    2) gap_decay float
        decrease the contribtuion of subsequences with large gaps (penalize non-contiguous)
    3) max_subsequence_length int 
        largest subsequence considered
    4) max_occurence_length int
        longest non-contiguous occurences of subsequences considered (max_occurence_length > max_subsequence_length)
    We calculate gradients for match_decay and gap_decay w.r.t kernel hyperparameters following Beck (2017)
    We recommend normalize = True to allow meaningful comparrison of strings of different length
    """
    def __init__(self, gap_decay=0.1, match_decay=0.9, max_subsequence_length=3,max_occurence_length=10,
                 alphabet = [], maxlen=0, normalize = True,batch_size=1000):
        super().__init__()
        # constrain kernel params to between 0 and 1
        logistic_gap = tfb.Chain([tfb.AffineScalar(shift=tf.cast(0,tf.float64),scale=tf.cast(1,tf.float64)),tfb.Sigmoid()])
        logisitc_match = tfb.Chain([tfb.AffineScalar(shift=tf.cast(0,tf.float64),scale=tf.cast(1,tf.float64)),tfb.Sigmoid()])
        self.gap_decay = Parameter(gap_decay, transform=logistic_gap,name="gap_decay")
        self.match_decay = Parameter(match_decay, transform=logisitc_match,name="match_decay")

        self.max_subsequence_length = max_subsequence_length
        self.max_occurence_length = max_occurence_length
        self.alphabet = alphabet
        self.maxlen = maxlen
        self.normalize = normalize
        self.batch_size = batch_size

        # build a lookup table of the alphabet to encode input strings
        self.table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(["PAD"]+alphabet),
                values=tf.constant(range(0,len(alphabet)+1)),),default_value=0)



    def K_diag(self, X):
        # Calc just the diagonal elements of a kernel matrix
        # check if string is not longer than max length
        # if  tf.reduce_max(tf.strings.length(X)) + 1 > 2 * self.maxlen:
        #             raise ValueError("An input string is longer that max-length so refit the kernel with a larger maxlen param")
        if self.normalize:
            # if normalizing then diagonal will just be ones
            return tf.cast(tf.fill(tf.shape(X)[:-1],1),tf.float64)
        else:
            # otherwise have to calc kernel elements
            # Turn our inputs into lists of integers using one-hot embedding
            # first split up strings and pad to fixed length and prep for gpu
            # pad until all same length
            X = tf.strings.split(tf.squeeze(X,1)).to_tensor("PAD",shape=[None,self.maxlen])
            X = self.table.lookup(X)
            return tf.reshape(self._diag_calculations(X),(-1,))

    def K(self, X, X2=None):
        print("Python execution")   ## This Line only Prints during Python Execution
        tf.print("Graph execution") ## This Line only Print during Graph Execution

        # check if symmetric (no provided X2), if so then only need to calc upper gram matrix 
        symmetric = True if (X2 is None) else False
        
        # check if input strings are longer than max allowed length
        # if  tf.reduce_max(tf.strings.length(X)) + 1 > 2 * self.maxlen:
        #             raise ValueError("An input string is longer that max-length so refit the kernel with a larger maxlen param")
        # if not symmetric:
        #     if  tf.reduce_max(tf.strings.length(X2)) + 1 > 2 * self.maxlen:
        #             raise ValueError("An input string is longer that max-length so refit the kernel with a larger maxlen param")        
        
        
        # Turn our inputs into lists of integers using one-hot embedding
        # first split up strings and pad to fixed length and prep for gpu
        # pad until all have length of self.maxlen
        X = tf.strings.split(tf.squeeze(X,1)).to_tensor("PAD",shape=[None,self.maxlen])
        X = self.table.lookup(X)
        if symmetric:
            X2 = X
        else:
            # pad until all have length of self.maxlen
            X2 = tf.strings.split(tf.squeeze(X2,1)).to_tensor("PAD",shape=[None,self.maxlen])
            X2 = self.table.lookup(X2)

        # get the decay tensor D
        D = self._precalc()

        # if needed pre calculate the values for kernel normalization
        if self.normalize:
            X_diag_Ks = tf.squeeze(self._diag_calculations(X))
            if not symmetric:
                X2_diag_Ks = tf.squeeze(self._diag_calculations(X2))
            

        # get indicies of all possible pairings from X and X2
        # this way allows maximum number of kernel calcs to be squished onto the GPU (rather than just doing individual rows of gram)
        indicies_2, indicies_1 = tf.meshgrid(tf.range(0, tf.shape(X2)[0]),tf.range(0, tf.shape(X)[0]))
        indicies = tf.concat([tf.reshape(indicies_1,(-1,1)),tf.reshape(indicies_2,(-1,1))],axis=1)
        # if symmetric then only calc upper matrix (fill in rest later)
        if symmetric:
            indicies = tf.boolean_mask(indicies,tf.greater_equal(indicies[:,1],indicies[:,0]))
        # make kernel calcs in batches
        k_results = tf.TensorArray(tf.float64, size=0, dynamic_size=True,infer_shape=False)
        num_batches = tf.math.ceil(tf.shape(indicies)[0]/self.batch_size)
        # iterate through batches
        for i in tf.range(tf.cast(tf.math.ceil(tf.shape(indicies)[0]/self.batch_size),dtype=tf.int32)):
            print("batch {}".(i))
            indicies_batch = indicies[self.batch_size*i:self.batch_size*(i+1)]
            X_batch = tf.gather(X,indicies_batch[:,0],axis=0)
            X2_batch = tf.gather(X2,indicies_batch[:,1],axis=0)
            k_results = k_results.write(k_results.size(), self._k(X_batch, X2_batch,D))
        



        # combine indivual kernel results
        k_results = tf.reshape(k_results.concat(),[1,-1])
        # put results into the right places in the gram matrix
        # if symmetric then only put in top triangle (inc diag)
        if symmetric:
            mask = tf.linalg.band_part(tf.ones((tf.shape(X)[0],tf.shape(X)[0]),dtype=tf.int64), 0, -1)
            non_zero = tf.not_equal(mask, tf.constant(0, dtype=tf.int64))
            indices = tf.where(non_zero) # Extracting the indices of upper triangle elements
            out = tf.SparseTensor(indices,tf.squeeze(k_results),dense_shape=tf.cast((tf.shape(X)[0],tf.shape(X)[0]),dtype=tf.int64))
            k_results = tf.sparse.to_dense(out)
            #add in mising elements
            k_results = k_results + tf.linalg.set_diag(tf.transpose(k_results),tf.zeros(tf.shape(X)[0],dtype=tf.float64))
        else:
            k_results = tf.reshape(k_results,[tf.shape(X)[0],tf.shape(X2)[0]])

        # normalize if required
        if self.normalize:
            if symmetric:
                norm = tf.sqrt(tf.tensordot(X_diag_Ks, X_diag_Ks,axes=0))
            else:
                norm = tf.sqrt(tf.tensordot(X_diag_Ks, X2_diag_Ks,axes=0))
            k_results = tf.divide(k_results, norm)
        return k_results



    def _diag_calculations(self, X):
            """
            Calculate the K(x,x) values first because
            they are used in normalization.
            This is pre-normalization (otherwise diag is just ones)
            """
            # Proceed with kernel matrix calculations in batches
            D = self._precalc()
            k_results = tf.TensorArray(tf.float64, size=0, dynamic_size=True,infer_shape=False)
            num_batches = tf.math.ceil(tf.shape(X)[0]/self.batch_size)
            # iterate through batches
            for i in tf.range(tf.cast(tf.math.ceil(tf.shape(X)[0]/self.batch_size),dtype=tf.int32)):
                X_batch = X[self.batch_size*i:self.batch_size*(i+1)]
                k_results = k_results.write(k_results.size(), self._k(X_batch, X_batch,D))

            # collect all batches
            k_results = tf.reshape(k_results.concat(),[1,-1])
            return k_results


    def _k(self, X1, X2, D):
        """
        Vectorized kernel calc.
        Following notation from Beck (2017), i.e have tensors S,D,Kpp,Kp
        Input is two tensors of shape (# strings , # characters)
        and we calc the pair-wise kernel calcs between the elements (i.e n kern calcs for two lists of length n)
        D is the tensor than unrolls the recursion and allows vecotrizaiton
        """

        # Strings will be represented as matrices of
        # one-hot embeddings and the similarity is just the dot product. (ie. checking for matches of characters)
        # turn into one-hot  i.e. shape (# strings, #characters+1, alphabet size)
        X1 = tf.one_hot(X1,len(self.alphabet)+1,dtype=tf.float64)
        X2 = tf.one_hot(X2,len(self.alphabet)+1,dtype=tf.float64)
        # remove the ones in the first column that encode the padding (i.e we dont want them to count as a match)
        paddings = tf.constant([[0, 0], [0, 0],[0,len(self.alphabet)]])
        X1 = X1 - tf.pad(tf.expand_dims(X1[:,:,0], 2),paddings,"CONSTANT")
        X2 = X2 - tf.pad(tf.expand_dims(X2[:,:,0], 2),paddings,"CONSTANT")
        # store squared match coef
        match_sq = tf.square(self.match_decay)
        # Make S: the similarity tensor of shape (# strings, #characters, # characters)
        S = tf.matmul( X1,tf.transpose(X2,perm=(0,2,1)))
        # Main loop, where Kp, Kpp values and gradients are calculated.
        Kp = tf.TensorArray(tf.float64, size=0, dynamic_size=True)
        Kp = Kp.write(Kp.size(), tf.ones(shape=tf.stack([tf.shape(X1)[0], self.maxlen,self.maxlen]), dtype=tf.float64))
        # calc subkernels for each subsequence length
        for i in tf.range(0,self.max_subsequence_length-1):
            aux = tf.multiply(S, Kp.read(i))
            aux = tf.reshape(aux, tf.stack([-1 , self.maxlen]))
            aux = tf.matmul(aux, D)
            aux = aux * match_sq
            aux = tf.reshape(aux, tf.stack([-1, self.maxlen, self.maxlen]))
            aux = tf.transpose(aux, perm=[0, 2, 1])
            aux = tf.reshape(aux, tf.stack([-1, self.maxlen]))
            aux = tf.matmul(aux, D)
            aux = tf.reshape(aux, tf.stack([-1, self.maxlen, self.maxlen]))
            Kp = Kp.write(Kp.size(),tf.transpose(aux, perm=[0, 2, 1]))
           
        # Final calculation. We gather all Kps 
        Kp = Kp.stack()
        # Get k
        aux = tf.multiply(S, Kp)
        aux = tf.reduce_sum(aux, 2)
        aux = tf.reduce_sum(aux, 2, keepdims=True)
        Ki = tf.multiply(aux, match_sq)
        Ki = tf.squeeze(Ki, [2])
        Ki = tf.expand_dims(Ki, 0)
        return tf.transpose(tf.reduce_sum(Ki,1))


    def _precalc(self):
        # pecalc D as required for every kernel calc
        # following notation from Beck (2017)

        # Make D: a upper triangular matrix over decay powers.
        tf_tril =  tf.linalg.band_part(tf.ones((self.maxlen,self.maxlen),dtype=tf.float64), -1, 0)
        power = [[0]*i+list(range(0,self.maxlen-i)) for i in range(1,self.maxlen)]+[[0]*self.maxlen]
        tf_power=tf.constant(power, dtype=tf.float64) + tf_tril
        #tf_tril = tf.transpose(tf_tril)-tf.eye(self.maxlen,dtype=tf.float64)
        tf_tril = tf.transpose(tf.linalg.band_part(tf.ones((self.maxlen,self.maxlen),dtype=tf.float64), self.max_occurence_length, 0))-tf.eye(self.maxlen,dtype=tf.float64)
        gaps = tf.fill([self.maxlen, self.maxlen], tf.squeeze(self.gap_decay))
        D = tf.pow(gaps*tf_tril, tf_power)
        return D











