import numpy as np
import tensorflow as tf

class TFStringKernel(object):
    """
    Code heavily borrowed from https://github.com/beckdaniel/flakes
    We make following changes
    1) Update to run on TF 2
    2) provide kernel normalization to make meaningful comparissons between strings of different lengths
    3) changed structure and conventions to match our Tree kernel implemenentation
    4) simplified to only allow one-hot encoidng of alphabet (i.e remove support for pre-trained embeddings)
    """
    def __init__(self, _gap_decay=1.0, _match_decay=1.0,
                  _order_coefs=[1.0], alphabet = [], maxlen=0,normalize=True,device=None):    
        self._gap_decay = _gap_decay
        self._match_decay = _match_decay
        self._order_coefs = _order_coefs
        self.alphabet = alphabet
        self.normalize = normalize
        # prepare one-hot representation of the alphabet to encode input strings
        try:
            self.embs, self.index = build_one_hot(self.alphabet)
        except Exception:
            raise Exception(" check input alphabet covers X")
        self.embs_dim = self.embs.shape[1]
        self.maxlen = maxlen
        # Not yet tested on GPU /gpu:0'
        self.device = device
        
    def Kdiag(self,X):   
        # input of form X = np.array([[s1],[s2],[s3]])
        # check if string is not longer than max length
        observed_maxlen = max([len(x[0].split(" ")) for x in X])
        if  observed_maxlen > self.maxlen:
            raise ValueError("An input string is longer that max-length so refit the kernel with a larger maxlen param")
        
        # if normalizing then diagonal will just be ones
        if self.normalize:
            return np.ones(X.shape[0])
        else:
            #otherwise have to calc
            # We turn our inputs into lists of integers using one-hot embedding
            # additional zero padding if their length is smaller than max-len
            X = np.array([[pad(encode_string(x[0].split(" "), self.index),self.maxlen)] for x in X])
            return self._diag_calculations(X)[0]

        
        
    def K(self, X, X2=None):
        # input of form X = np.array([[s1],[s2],[s3]])
        # check if symmetric (no provided X2), if so then only need to calc upper gram matrix 
        symmetric = True if (X2 is None) else False
        
        # check if input strings are longer than max allowed length
        observed_maxlen = max([len(x[0].split(" ")) for x in X])
        if not symmetric:
            observed_maxlen_2 = max([len(x[0].split(" ")) for x in X])
            observed_maxlen = max(observed_maxlen,observed_maxlen_2)
        if  observed_maxlen > self.maxlen:
            raise ValueError("An input string is longer that max-length so refit the kernel with a larger maxlen param")
        
        # Turn our inputs into lists of integers using one-hot embedding
        # additional zero padding if their length is smaller than max-len
        X = np.array([[pad(encode_string(x[0].split(" "), self.index),self.maxlen)] for x in X])
        if symmetric:
            X2 = X
        else:
            X2 = np.array([[pad(encode_string(x[0].split(" "), self.index),self.maxlen)] for x in X2])
        

        #if needed calculate the values needed for normalization
        if self.normalize:
            X_diag_Ks, X_diag_gap_grads, X_diag_match_grads, X_diag_coef_grads = self._diag_calculations(X)
            if not symmetric:
                X2_diag_Ks, X2_diag_gap_grads, X2_diag_match_grads, X2_diag_coef_grads = self._diag_calculations(X2)
            

        # Initialize return values
        k_results = np.zeros(shape=(len(X), len(X2)))
        gap_grads = np.zeros(shape=(len(X), len(X2)))
        match_grads = np.zeros(shape=(len(X), len(X2)))
        coef_grads = np.zeros(shape=(len(X), len(X2), len(self._order_coefs)))
        
        # All set up. Proceed with kernel matrix calculations
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X2):
                # if symmetric then only need to actually calc the upper gram matrix
                if symmetric and (j < i):
                    k_results[i, j] = k_results[j, i]
                    gap_grads[i, j] = gap_grads[j, i]
                    match_grads[i, j] = match_grads[j, i]
                    coef_grads[i, j,:] = coef_grads[j, i,:]
                else:
                    k_result, gap_grad, match_grad, coef_grad = self._k(x1[0], x2[0])

                    # Normalize kernel and gradients if required
                    if self.normalize:
                        if symmetric:
                            k_result_norm, gap_grad_norm, match_grad_norm, coef_grad_norm = self._normalize(k_result, gap_grad, match_grad,coef_grad,
                                                                                            X_diag_Ks[i], X_diag_Ks[j],
                                                                                            X_diag_gap_grads[i], X_diag_match_grads[i],X_diag_coef_grads[i],
                                                                                            X_diag_gap_grads[j], X_diag_match_grads[j],X_diag_coef_grads[j])
                        else:
                            k_result_norm, gap_grad_norm, match_grad_norm, coef_grad_norm = self._normalize(k_result, gap_grad, match_grad,coef_grad,
                                                                                            X_diag_Ks[i], X2_diag_Ks[j],
                                                                                            X_diag_gap_grads[i], X_diag_match_grads[i],X_diag_coef_grads[i],
                                                                                            X2_diag_gap_grads[j], X2_diag_match_grads[j],X2_diag_coef_grads[j])
                        k_results[i, j] = k_result_norm
                        gap_grads[i, j] = gap_grad_norm
                        match_grads[i, j] = match_grad_norm
                        coef_grads[i, j] = np.array(coef_grad_norm) 
                    else:
                        k_results[i, j] = k_result
                        gap_grads[i, j] = gap_grad
                        match_grads[i, j] = match_grad
                        coef_grads[i, j] = np.array(coef_grad)    
        return k_results, gap_grads, match_grads, coef_grads


    def _normalize(self, K_result, gap_grads, match_grads, coef_grads,diag_Ks_i,
                    diag_Ks_j, diag_gap_grads_i, diag_match_grads_i, diag_coef_grads_i,
                    diag_gap_grads_j, diag_match_grads_j, diag_coef_grads_j,):
        """
        Normalize the kernel and kernel derivatives.
        Following the derivation of Beck (2015)
        """
        norm = diag_Ks_i * diag_Ks_j
        sqrt_norm = np.sqrt(norm)
        K_norm = K_result / sqrt_norm
        
                
        diff_gap = ((diag_gap_grads_i * diag_Ks_j) +
                       (diag_Ks_i * diag_gap_grads_j))
        diff_gap /= 2 * norm
        gap_grads_norm = ((gap_grads / sqrt_norm) -
                        (K_norm * diff_gap))
        
        diff_match = ((diag_match_grads_i * diag_Ks_j) +
                       (diag_Ks_i * diag_match_grads_j))
        diff_match /= 2 * norm

        match_grads_norm = ((match_grads / sqrt_norm) -
                        (K_norm * diff_match))
        

        diff_coef = ((diag_coef_grads_i * diag_Ks_j) +
                       (diag_Ks_i * diag_coef_grads_j))

        diff_coef /= 2 * norm

        coef_grads_norm = ((coef_grads / sqrt_norm) -
                        (K_norm * diff_coef))

        return K_norm, gap_grads_norm, match_grads_norm, coef_grads_norm
        


    def _diag_calculations(self, X):
        """
        Calculate the K(x,x) values first because
        they are used in normalization.
        This is pre-normalization (otherwise diag is just ones)
        This function is not to be called directly, as requires preprocessing on X
        """
        # initialize return values
        k_result = np.zeros(shape=(len(X)))
        gap_grads = np.zeros(shape=(len(X)))
        match_grads = np.zeros(shape=(len(X)))
        coef_grads = np.zeros(shape=(len(X), len(self._order_coefs)))
        
        # All set up. Proceed with kernel matrix calculations
        for i, x1 in enumerate(X):
            result = self._k(x1[0], x1[0])
            k_result[i] = result[0]
            gap_grads[i] = result[1]
            match_grads[i] = result[2]
            coef_grads[i] = np.array(result[3])
        return (k_result,gap_grads,match_grads,coef_grads)
        

    
    
    def _k(self, s1, s2):
        """
        TF code for vecotrized kernel calc.
        Following notation from Beck (2017), i.e have tensors S,D,Kpp,Kp
        
        Input is two lists (of length maxlen) of integers represents each character in the alphabet 
        calc kernel between these two string representations. 
        """

        # init
        tf_embs = tf.constant(self.embs, dtype=tf.float64, name='embs')
        tf_s1 = tf.convert_to_tensor(s1, dtype=tf.int32)
        tf_s2 = tf.convert_to_tensor(s2, dtype=tf.int32)
        tf_gap_decay = tf.constant(self._gap_decay,dtype=tf.float64)
        tf_match_decay = tf.constant(self._match_decay,dtype=tf.float64)
        tf_order_coefs = tf.convert_to_tensor(self._order_coefs, dtype=tf.float64)

        # keep track of required gradients
        with tf.GradientTape(persistent=True) as t:
            t.watch(tf_gap_decay)
            t.watch(tf_match_decay)
            t.watch(tf_order_coefs)

            # Strings will be represented as matrices of
            # embeddings and the similarity is just
            # the dot product. 
            # Make S: the similarity matrix
            S = tf.matmul(tf.gather(tf_embs, tf_s1), tf.transpose(tf.gather(tf_embs, tf_s2)))

            # Make D: a upper triangular matrix over decay powers.
            # build it in vectorized numpy because tf doesnt allow item assignment
            power = np.ones((self.maxlen, self.maxlen))
            tril = np.zeros((self.maxlen, self.maxlen))
            i1, i2 = np.indices(power.shape)
            for k in range(self.maxlen - 1):
                power[i2-k-1 == i1] = k
                tril[i2-k-1 == i1] = 1.0

            # now convert to tf
            tf_power = tf.constant(power, dtype=tf.float64)
            tf_tril = tf.constant(tril, dtype=tf.float64)
            gaps = tf.fill([self.maxlen, self.maxlen], tf_gap_decay)
            D = tf.pow(gaps*tf_tril, tf_power)

            # Main loop, where Kp, Kpp values are calculated.
            Kp = []
            Kp.append(tf.ones(shape=(self.maxlen, self.maxlen), dtype="float64"))
            for i in range(len(tf_order_coefs)-1):
                aux1 = S * Kp[i]
                aux2 = tf.matmul(aux1, D)
                Kpp = tf.square(tf_match_decay) * aux2
                Kp.append(tf.transpose(tf.matmul(tf.transpose(Kpp), D)))
            final_Kp = tf.stack(Kp)

            # Final calculation. We gather all Kps and
            # multiply then by their coeficients.
            mul1 = S * final_Kp[:len(tf_order_coefs), :, :]
            sum1 = tf.reduce_sum(mul1, 1)
            Ki = tf.reduce_sum(sum1, 1, keepdims=True) * tf.square(tf_match_decay)
            k = tf.matmul(tf.reshape(tf_order_coefs,(1,-1)), Ki)

        # auto diff to get gradients    
        gap_grads = t.gradient(k, tf_gap_decay).numpy()
        match_grads = t.gradient(k, tf_match_decay).numpy()
        coef_grads = t.gradient(k, tf_order_coefs).numpy()
        #return scalars apart from coef_grads (1-d np)
        result = k.numpy()[0][0]
        return result, gap_grads, match_grads, coef_grads

# helper functions to perform operations on input strings

def pad(s, length):
    #pad out input strings to our maxlen
    new_s = np.zeros(length)
    new_s[:len(s)] = s
    return new_s

def encode_string(s, index):
    """
    Transform a string in a list of integers.
    The ints correspond to indices in an
    embeddings matrix.
    """
    return [index[symbol] for symbol in s]

def build_one_hot(alphabet):
    """
    Build one-hot encodings for a given alphabet.
    """
    dim = len(alphabet)
    embs = np.zeros((dim+1, dim))
    index = {}
    for i, symbol in enumerate(alphabet):
        embs[i+1, i] = 1.0
        index[symbol] = i+1
    return embs, index