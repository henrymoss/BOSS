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
    def __init__(self, active_dims=[0],gap_decay=0.1, match_decay=0.9, variance=1,max_subsequence_length=3,max_occurence_length=10,
                 alphabet = [], maxlen=0, normalize = True,batch_size=1000):
        super().__init__(active_dims=active_dims)
        # constrain kernel params to between 0 and 1
        logistic_gap = tfb.Chain([tfb.AffineScalar(shift=tf.cast(0,tf.float64),scale=tf.cast(1,tf.float64)),tfb.Sigmoid()])
        logisitc_match = tfb.Chain([tfb.AffineScalar(shift=tf.cast(0,tf.float64),scale=tf.cast(1,tf.float64)),tfb.Sigmoid()])
        self.gap_decay_param = Parameter(gap_decay, transform=logistic_gap,name="gap_decay")
        self.match_decay_param = Parameter(match_decay, transform=logisitc_match,name="match_decay")
        self.variance_param = Parameter(variance, transform=positive(),name="variance")
        self.max_subsequence_length = max_subsequence_length
        self.max_occurence_length = max_occurence_length
        self.alphabet = alphabet
        self.maxlen = maxlen
        self.normalize = normalize
        self.batch_size = batch_size
        self.symmetric = False

        # use will use copies of the kernel params to stop building expensive computation graph
        # we instead efficientely calculate gradients using dynamic programming
        # These params are updated at every call to K and K_diag (to check if parameters have been updated)
        self.match_decay = self.match_decay_param.numpy()
        self.gap_decay = self.gap_decay_param.numpy()
        self.variance = self.variance_param.numpy()

        # initialize helful construction matricies to be lazily computed once needed
        self.D = None
        self.dD_dgap = None


        # build a lookup table of the alphabet to encode input strings
        self.table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(["PAD"]+alphabet),
                values=tf.constant(range(0,len(alphabet)+1)),),default_value=0)


    def K_diag(self, X):
        r"""
        Calc just the diagonal elements of a kernel matrix
        """

        # check if string is not longer than max length
        if  tf.reduce_max(tf.strings.length(X)) + 1 > 2 * self.maxlen:
                    raise ValueError("An input string is longer that max-length so refit the kernel with a larger maxlen param")
        
        if self.normalize:
            # if normalizing then diagonal will just be ones
            return tf.cast(tf.fill(tf.shape(X)[:-1],1),tf.float64)
        else:
            # otherwise have to calc kernel elements
            # Turn inputs into lists of integers using one-hot embedding and pad until all same length
            X = tf.strings.split(tf.squeeze(X,1)).to_tensor("PAD",shape=[None,self.maxlen])
            X = self.table.lookup(X)

            # prep required quantities and check kernel parameters
            self._precalc()

            # Proceed with kernel matrix calculations in batches
            k_results = tf.TensorArray(tf.float64, size=0, dynamic_size=True,infer_shape=False)

            num_batches = tf.math.ceil(tf.shape(X)[0]/self.batch_size)
            # iterate through batches
            for i in tf.range(tf.cast(tf.math.ceil(tf.shape(X)[0]/self.batch_size),dtype=tf.int32)):
                X_batch = X[self.batch_size*i:self.batch_size*(i+1)]
                k_results = k_results.write(k_results.size(), self._k(X_batch, X_batch))

            # collect all batches
            return tf.reshape(k_results.concat(),(-1,))


    def K(self,X,X2=None):
        r"""
        Now we calculate the kernel values and kernel gradients
        Efficientely calculating kernel gradients requires dynamic programming 
        and so we 'turn off' autograd and calculate manually

        We currentely only bother calculating the kernel gradients for gram matricies
        i.e (when X=X2) as required when fitting the model.
        For predictions (where X != X2) we do not calculate gradients
        """

        # we currentely only bother calculating the kernel gradients for gram matricies
        # i.e (when X=X2) as required when fitting the model, denoted as symmetric = True
        # For predictions (where X != X2) we do not calculate gradients

        if X2 is None:
            self.symmetric = True
            k_results = self.K_calc(X,X)
        else:
            self.symmetric = False
            k_results = self.K_calc(X,X2)

        return k_results



    def _precalc(self):
        r"""
        Update stored kernel params (incase they have changed)
        and precalc D and dD_dgap as required for kernel calcs
        following notation from Beck (2017)
        """
        self.match_decay = self.match_decay_param.numpy()
        self.gap_decay = self.gap_decay_param.numpy()
        self.variance = self.variance_param.numpy()

        tril =  tf.linalg.band_part(tf.ones((self.maxlen,self.maxlen),dtype=tf.float64), -1, 0)
        # get upper triangle matrix of increasing intergers
        values = tf.TensorArray(tf.int32, size= self.maxlen)
        for i in tf.range(self.maxlen):
            values = values.write(i,tf.range(-i-1,self.maxlen-1-i)) 
        power = tf.cast(values.stack(),tf.float64)
        values.close()
        power = tf.linalg.band_part(power, 0, -1) - tf.linalg.band_part(power, 0, 0) + tril
        tril = tf.transpose(tf.linalg.band_part(tf.ones((self.maxlen,self.maxlen),dtype=tf.float64), self.max_occurence_length, 0))-tf.eye(self.maxlen,dtype=tf.float64)
        gaps = tf.fill([self.maxlen, self.maxlen],self.gap_decay)
        
        self.D = tf.pow(gaps*tril, power)
        self.dD_dgap = tf.pow((tril * gaps), (power - 1.0)) * tril * power



    @tf.custom_gradient
    def K_calc(self, X, X2):
        r"""
        Calc the elements of the kernel matrix (and gradients if symmetric)
        """
        
        # check if input strings are longer than max allowed length
        if  (tf.reduce_max(tf.strings.length(X)) + 1 > 2 * self.maxlen) or (tf.reduce_max(tf.strings.length(X2)) + 1 > 2 * self.maxlen):
                    raise ValueError("An input string is longer that max-length so refit the kernel with a larger maxlen param")
     
        # Turn our inputs into lists of integers using one-hot embedding
        # first split up strings and pad to fixed length and prep for gpu
        # pad until all have length of self.maxlen
        X = tf.strings.split(tf.squeeze(X,1)).to_tensor("PAD",shape=[None,self.maxlen])
        X = self.table.lookup(X)
        if self.symmetric:
            X2 = X
        else:
            # pad until all have length of self.maxlen
            X2 = tf.strings.split(tf.squeeze(X2,1)).to_tensor("PAD",shape=[None,self.maxlen])
            X2 = self.table.lookup(X2)

        # get the decay tensors D and dD_dgap
        self._precalc()

        # get indicies of all possible pairings from X and X2
        # this way allows maximum number of kernel calcs to be squished onto the GPU (rather than just doing individual rows of gram)
        indicies_2, indicies_1 = tf.meshgrid(tf.range(0, tf.shape(X2)[0]),tf.range(0, tf.shape(X)[0]))
        indicies = tf.concat([tf.reshape(indicies_1,(-1,1)),tf.reshape(indicies_2,(-1,1))],axis=1)
        # if symmetric then only calc upper matrix (fill in rest later)
        if self.symmetric:
            indicies = tf.boolean_mask(indicies,tf.greater_equal(indicies[:,1],indicies[:,0]))
        # make kernel calcs in batches
        num_batches = tf.math.ceil(tf.shape(indicies)[0]/self.batch_size)
        # iterate through batches
        
        if self.symmetric:
            k_results = tf.TensorArray(tf.float64, size=0, dynamic_size=True,infer_shape=False)
            gap_grads = tf.TensorArray(tf.float64, size=0, dynamic_size=True,infer_shape=False)
            match_grads = tf.TensorArray(tf.float64, size=0, dynamic_size=True,infer_shape=False)
            for i in tf.range(tf.cast(tf.math.ceil(tf.shape(indicies)[0]/self.batch_size),dtype=tf.int32)):
                indicies_batch = indicies[self.batch_size*i:self.batch_size*(i+1)]
                X_batch = tf.gather(X,indicies_batch[:,0],axis=0)
                X2_batch = tf.gather(X2,indicies_batch[:,1],axis=0)
                results = self._k_grads(X_batch, X2_batch)
                k_results = k_results.write(k_results.size(), results[0])
                gap_grads = gap_grads.write(gap_grads.size(),results[1])
                match_grads = match_grads.write(match_grads.size(),results[2])
            # combine indivual kernel results
            k_results = tf.reshape(k_results.concat(),[1,-1])
            gap_grads = tf.reshape(gap_grads.concat(),[1,-1])
            match_grads = tf.reshape(match_grads.concat(),[1,-1])
        else:
            k_results = tf.TensorArray(tf.float64, size=0, dynamic_size=True,infer_shape=False)
            for i in tf.range(tf.cast(tf.math.ceil(tf.shape(indicies)[0]/self.batch_size),dtype=tf.int32)):
                indicies_batch = indicies[self.batch_size*i:self.batch_size*(i+1)]
                X_batch = tf.gather(X,indicies_batch[:,0],axis=0)
                X2_batch = tf.gather(X2,indicies_batch[:,1],axis=0)
                k_results = k_results.write(k_results.size(), self._k(X_batch, X2_batch))
            # combine indivual kernel results
            k_results = tf.reshape(k_results.concat(),[1,-1])



        # put results into the right places in the gram matrix
        # if symmetric then only put in top triangle (inc diag)
        if self.symmetric:
            mask = tf.linalg.band_part(tf.ones((tf.shape(X)[0],tf.shape(X)[0]),dtype=tf.int64), 0, -1)
            non_zero = tf.not_equal(mask, tf.constant(0, dtype=tf.int64))
            indices = tf.where(non_zero) # Extracting the indices of upper triangle elements
            out = tf.SparseTensor(indices,tf.squeeze(k_results),dense_shape=tf.cast((tf.shape(X)[0],tf.shape(X)[0]),dtype=tf.int64))
            k_results = tf.sparse.to_dense(out)
            out = tf.SparseTensor(indices,tf.squeeze(gap_grads),dense_shape=tf.cast((tf.shape(X)[0],tf.shape(X)[0]),dtype=tf.int64))
            gap_grads = tf.sparse.to_dense(out)
            out = tf.SparseTensor(indices,tf.squeeze(match_grads),dense_shape=tf.cast((tf.shape(X)[0],tf.shape(X)[0]),dtype=tf.int64))
            match_grads = tf.sparse.to_dense(out)

            #add in mising elements (lower diagonal)
            k_results = k_results + tf.linalg.set_diag(tf.transpose(k_results),tf.zeros(tf.shape(X)[0],dtype=tf.float64))
            gap_grads = gap_grads + tf.linalg.set_diag(tf.transpose(gap_grads),tf.zeros(tf.shape(X)[0],dtype=tf.float64))
            match_grads = match_grads + tf.linalg.set_diag(tf.transpose(match_grads),tf.zeros(tf.shape(X)[0],dtype=tf.float64))
        else:
            k_results = tf.reshape(k_results,[tf.shape(X)[0],tf.shape(X2)[0]])


        # normalize if required
        if self.normalize:
            if self.symmetric:
                # if symmetric then can extract normalization terms from gram
                X_diag_Ks = tf.linalg.diag_part(k_results)
                X_diag_gap_grads = tf.linalg.diag_part(gap_grads)
                X_diag_match_grads = tf.linalg.diag_part(match_grads)

                # norm for kernel entries
                norm = tf.tensordot(X_diag_Ks, X_diag_Ks,axes=0)
                k_results = tf.divide(k_results, tf.sqrt(norm))
                # norm for gap_decay and match_decay grads
                diff_gap = tf.divide(tf.tensordot(X_diag_gap_grads,X_diag_Ks,axes=0) + tf.tensordot(X_diag_Ks,X_diag_gap_grads,axes=0),2 * norm)
                diff_match = tf.divide(tf.tensordot(X_diag_match_grads,X_diag_Ks,axes=0) + tf.tensordot(X_diag_Ks,X_diag_match_grads,axes=0),2 * norm)
                gap_grads= tf.divide(gap_grads, tf.sqrt(norm)) - tf.multiply(k_results,diff_gap)
                match_grads = tf.divide(match_grads, tf.sqrt(norm)) - tf.multiply(k_results,diff_match)
            else:
                # if not symmetric then need to calculate some extra kernel calcs
                # get diagonal kernel calcs for X1
                X_diag_Ks = tf.TensorArray(tf.float64, size=0, dynamic_size=True,infer_shape=False)
                num_batches = tf.math.ceil(tf.shape(X)[0]/self.batch_size)
                # iterate through batches
                for i in tf.range(tf.cast(tf.math.ceil(tf.shape(X)[0]/self.batch_size),dtype=tf.int32)):
                    X_batch = X[self.batch_size*i:self.batch_size*(i+1)]
                    X_diag_Ks = X_diag_Ks.write(X_diag_Ks.size(), self._k(X_batch, X_batch))
                # collect up all batches
                X_diag_Ks = tf.reshape(X_diag_Ks.concat(),(-1,))
        
                # get diagonal kernel calcs for X2
                X2_diag_Ks = tf.TensorArray(tf.float64, size=0, dynamic_size=True,infer_shape=False)
                num_batches = tf.math.ceil(tf.shape(X2)[0]/self.batch_size)
                # iterate through batches
                for i in tf.range(tf.cast(tf.math.ceil(tf.shape(X2)[0]/self.batch_size),dtype=tf.int32)):
                    X2_batch = X2[self.batch_size*i:self.batch_size*(i+1)]
                    X2_diag_Ks = X2_diag_Ks.write(X2_diag_Ks.size(), self._k(X2_batch, X2_batch))
                # collect up all batches
                X2_diag_Ks = tf.reshape(X2_diag_Ks.concat(),(-1,))


                # norm for kernel entries
                norm = tf.tensordot(X_diag_Ks, X2_diag_Ks,axes=0)
                k_results = tf.divide(k_results, tf.sqrt(norm))

        # finally scale with variance
        k_final = tf.multiply(k_results,self.variance)


        def grad(*dy, variables=None):
            print(variables)
            if self.symmetric:
                gradient = [tf.reduce_sum(tf.multiply(dy,gap_grads)) , tf.reduce_sum(tf.multiply(dy,match_grads)),tf.reduce_sum(tf.multiply(dy,k_results))]
                return ((None,None),gradient)
            else:
                return ((None,None),[None,None,None])

        return k_final, grad



    def _k_grads(self, X1, X2):
        r"""
        Vectorized kernel calc and kernel grad calc.
        Following notation from Beck (2017), i.e have tensors S,D,Kpp,Kp
        Input is two tensors of shape (# strings , # characters)
        and we calc the pair-wise kernel calcs between the elements (i.e n kern calcs for two lists of length n)
        D is the tensor than unrolls the recursion and allows vecotrizaiton
        """

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
        Kp = tf.TensorArray(tf.float64, size=0, dynamic_size=True,clear_after_read=False)
        dKp_dgap = tf.TensorArray(tf.float64, size=0, dynamic_size=True,clear_after_read=False)
        dKp_dmatch = tf.TensorArray(tf.float64, size=0, dynamic_size=True,clear_after_read=False)
        Kp = Kp.write(Kp.size(), tf.ones(shape=tf.stack([tf.shape(X1)[0], self.maxlen,self.maxlen]), dtype=tf.float64))
        dKp_dgap = dKp_dgap.write(dKp_dgap.size(), tf.zeros(shape=tf.stack([tf.shape(X1)[0], self.maxlen,self.maxlen]), dtype=tf.float64))
        dKp_dmatch = dKp_dmatch.write(dKp_dmatch.size(), tf.zeros(shape=tf.stack([tf.shape(X1)[0], self.maxlen,self.maxlen]), dtype=tf.float64))
        # calc subkernels for each subsequence length
        for i in tf.range(0,self.max_subsequence_length-1):
            aux = tf.multiply(S, Kp.read(i))
            aux1 = tf.reshape(aux, tf.stack([-1 , self.maxlen]))
            aux2 = tf.matmul(aux1, self.D)
            aux = aux2 * match_sq
            aux = tf.reshape(aux, tf.stack([-1, self.maxlen, self.maxlen]))
            aux = tf.transpose(aux, perm=[0, 2, 1])
            aux3 = tf.reshape(aux, tf.stack([-1, self.maxlen]))
            aux = tf.matmul(aux3, self.D)
            aux = tf.reshape(aux, tf.stack([-1, self.maxlen, self.maxlen]))
            Kp = Kp.write(Kp.size(),tf.transpose(aux, perm=[0, 2, 1]))

            aux = tf.multiply(S, dKp_dgap.read(i))
            aux = tf.reshape(aux, tf.stack([-1,self.maxlen]))
            aux = tf.matmul(aux, self.D) + tf.matmul(aux1, self.dD_dgap)
            aux = aux * match_sq
            aux = tf.reshape(aux, tf.stack([-1,self.maxlen,self.maxlen]))
            aux = tf.transpose(aux, perm=[0, 2, 1])
            aux = tf.reshape(aux, tf.stack([-1,self.maxlen]))
            aux = tf.matmul(aux, self.D) + tf.matmul(aux3, self.dD_dgap)
            aux = tf.reshape(aux, tf.stack([-1,self.maxlen,self.maxlen]))
            dKp_dgap = dKp_dgap.write(dKp_dgap.size(),tf.transpose(aux, perm=[0, 2, 1]))


            aux = tf.multiply(S, dKp_dmatch.read(i))
            aux = tf.reshape(aux, tf.stack([-1,self.maxlen]))
            aux = tf.matmul(aux, self.D)
            aux = (aux * match_sq) + (2 * self.match_decay * aux2)
            aux = tf.reshape(aux, tf.stack([-1,self.maxlen,self.maxlen]))
            aux = tf.transpose(aux, perm=[0, 2, 1])
            aux = tf.reshape(aux, tf.stack([-1,self.maxlen]))
            aux = tf.matmul(aux, self.D)
            aux = tf.reshape(aux, tf.stack([-1,self.maxlen,self.maxlen]))
            dKp_dmatch = dKp_dmatch.write(dKp_dmatch.size(),tf.transpose(aux, perm=[0, 2, 1]))

           
        # Final calculation. We gather all Kps 
        Kp_stacked = Kp.stack()
        Kp.close()
        dKp_dgap_stacked = dKp_dgap.stack()
        dKp_dgap.close()
        dKp_dmatch_stacked = dKp_dmatch.stack()
        dKp_dmatch.close()


        # Get k
        aux = tf.multiply(S, Kp_stacked)
        aux = tf.reduce_sum(aux, 2)
        sum2 = tf.reduce_sum(aux, 2, keepdims=True)
        Ki = tf.multiply(sum2, match_sq)
        Ki = tf.squeeze(Ki, [2])
        Ki = tf.expand_dims(Ki, 0)
        k = tf.transpose(tf.reduce_sum(Ki,1))

        # get gap decay grads
        aux = tf.multiply(S, dKp_dgap_stacked)
        aux = tf.reduce_sum(aux, 2)
        aux = tf.reduce_sum(aux, 2, keepdims=True)
        aux = tf.multiply(aux, match_sq)
        aux = tf.squeeze(aux, [2])
        aux = tf.expand_dims(aux, 0)
        dk_dgap = tf.transpose(tf.reduce_sum(aux,1))

        # get match decay grads
        aux = tf.multiply(S, dKp_dmatch_stacked)
        aux = tf.reduce_sum(aux, 2)
        aux = tf.reduce_sum(aux, 2, keepdims=True)
        aux = tf.multiply(aux, match_sq) + (2 * self.match_decay * sum2)
        aux = tf.squeeze(aux, [2])
        aux = tf.expand_dims(aux, 0)
        dk_dmatch = tf.transpose(tf.reduce_sum(aux,1))

        return k, dk_dgap, dk_dmatch




    def _k(self, X1, X2):
        r"""
        Vectorized kernel calc.
        Following notation from Beck (2017), i.e have tensors S,D,Kpp,Kp
        Input is two tensors of shape (# strings , # characters)
        and we calc the pair-wise kernel calcs between the elements (i.e n kern calcs for two lists of length n)
        D is the tensor than unrolls the recursion and allows vecotrizaiton
        """

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
        Kp = tf.TensorArray(tf.float64, size=0, dynamic_size=True,clear_after_read=False)
        Kp = Kp.write(Kp.size(), tf.ones(shape=tf.stack([tf.shape(X1)[0], self.maxlen,self.maxlen]), dtype=tf.float64))
        # calc subkernels for each subsequence length
        for i in tf.range(0,self.max_subsequence_length-1):
            aux = tf.multiply(S, Kp.read(i))
            aux1 = tf.reshape(aux, tf.stack([-1 , self.maxlen]))
            aux2 = tf.matmul(aux1, self.D)
            aux = aux2 * match_sq
            aux = tf.reshape(aux, tf.stack([-1, self.maxlen, self.maxlen]))
            aux = tf.transpose(aux, perm=[0, 2, 1])
            aux3 = tf.reshape(aux, tf.stack([-1, self.maxlen]))
            aux = tf.matmul(aux3, self.D)
            aux = tf.reshape(aux, tf.stack([-1, self.maxlen, self.maxlen]))
            Kp = Kp.write(Kp.size(),tf.transpose(aux, perm=[0, 2, 1]))
           
        # Final calculation. We gather all Kps 
        Kp_stacked = Kp.stack()
        Kp.close()
    

        # Get k
        aux = tf.multiply(S, Kp_stacked)
        aux = tf.reduce_sum(aux, 2)
        sum2 = tf.reduce_sum(aux, 2, keepdims=True)
        Ki = tf.multiply(sum2, match_sq)
        Ki = tf.squeeze(Ki, [2])
        Ki = tf.expand_dims(Ki, 0)
        k = tf.transpose(tf.reduce_sum(Ki,1))


        return k





