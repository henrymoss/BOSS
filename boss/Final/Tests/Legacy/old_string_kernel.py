import numpy as np

#THIS IS OLD CODE FROM https://github.com/beckdaniel/flakes/blob/master/flakes/string/sk_util.py
# IT IS INCLUDED TO SANITY CHECK OUR NEW IMPLEMENTATIONS

class OLDStringKernel(object):
    """
    A general class for String Kernels. Default mode is
    TensorFlow mode but numpy and non-vectorized implementations
    are also available. The last two should be used only for
    testing and debugging, TF is generally faster, even in
    CPU-only environments.
    
    The parameterization is based on:
    
    Cancedda et. al (2003) "Word-Sequence Kernels" JMLR
    with two different decay parameters: one for gaps
    and another one for symbol matchings. There is 
    also a list of order coeficients which weights
    different n-grams orders. The overall order
    is implicitly obtained by the size of this list.
    This is *not* the symbol-dependent version.
    
    :param gap_decay: decay for symbols gaps, defaults to 1.0
    :param match_decay: decay for symbols matches, defaults to 1.0
    :param order_coefs: list of coefficients for different ngram
    orders, defaults to [1.0]
    :param mode: inner kernel implementation, defaults to TF
    :param device: where to run the inner kernel calculation,
    in TF nomenclature (only used if in TF mode).
    """

    def __init__(self, gap_decay=1.0, match_decay=1.0,
                 order_coefs=[1.0], alphabet=None):
        embs, self.index = build_one_hot(alphabet)
        self.gap_decay = gap_decay
        self.match_decay = match_decay
        self.order_coefs = order_coefs
        self.wrapper = 'none'
        self._implementation = NumpyStringKernel(embs=embs, sim='dot')


    @property
    def order(self):
        """
        Kernel ngram order, defined implicitly.
        """
        return len(self.order_coefs)

    def _get_params(self):
        return [self.gap_decay, self.match_decay, 
                self.order_coefs]

    def K(self, X, X2=None, diag=False):
        """
        Calculate the Gram matrix over two lists of strings. The
        underlying method used for kernel calculation depends
        on self.mode (slow, numpy or TF). 
        """
        # Symmetry check to ensure that we only calculate
        # the lower diagonal.
        if X2 is None and not diag:
            X2 = X
            gram = True
        else:
            gram = False

        # This can also be calculated for single elements but
        # we need to explicitly convert to lists before any
        # processing
        if not (isinstance(X, list) or isinstance(X, np.ndarray)):
            X = np.array([[X]])
        if not (isinstance(X2, list) or isinstance(X2, np.ndarray)):
            X2 = np.array([[X2]])

        # Now we turn our inputs into lists of integers using the
        # index
        if self.index is not None:
            # If index is none we assume inputs are already
            # encoded in integer lists.
            X = np.array([[encode_string(x[0], self.index)] for x in X])
            #if not diag:
            X2 = np.array([[encode_string(x2[0], self.index)] for x2 in X2])
        #print self.index

        params = self._get_params()
        #print X
        #print X2
        #print diag
        result = self._implementation.K(X, X2, gram=gram, params=params, diag=diag)
        k_result = result[0]

        self.gap_grads = result[1]
        self.match_grads = result[2]
        self.coef_grads = result[3]
        if self.wrapper != 'none':
            self.var_grads = result[4]
        return k_result

class NumpyStringKernel(object):
    """
    A vectorized string kernel implementation.
    It is faster than the naive version but
    slower than TensorFlow versions. Also
    kept for documentary and testing purposes.
    """
    def __init__(self, embs, sim='dot'):
        self.embs = embs
        self.embs_dim = embs.shape[1]
        if sim == 'arccosine':
            self.sim = self._arccosine
            self.norms = np.sqrt(np.sum(pow(embs, 2), 1, keepdims=True))
        elif sim == 'dot':
            self.sim = self._dot

    def _k(self, s1, s2, params):
        """
        The actual string kernel calculation. It also
        calculates gradients with respect to the
        multiple hyperparameters.
        """
        n = len(s1)
        m = len(s2)
        gap = params[0]
        match = params[1]
        coefs = params[2]
        order = len(coefs)

        # Transform inputs into embedding matrices
        #embs1 = self.embs[s1]
        #embs2 = self.embs[s2]
        
        # Triangular matrix over decay powers
        maxlen = max(n, m)
        power = np.ones((maxlen, maxlen))
        tril = np.zeros((maxlen, maxlen))
        i1, i2 = np.indices(power.shape)
        for k in range(maxlen - 1):
            power[i2-k-1 == i1] = k
            tril[i2-k-1 == i1] = 1.0
        gaps = np.ones((maxlen, maxlen)) * gap
        D = (gaps * tril) ** power
        dD_dgap = ((gaps * tril) ** (power - 1.0)) * tril * power

        # Store sim(j, k) values
        S = self.sim(s1, s2)
        #print S

        # Initializing auxiliary variables
        Kp = np.ones(shape=(order, n, m))
        dKp_dgap = np.zeros(shape=(order, n, m))
        dKp_dmatch = np.zeros(shape=(order, n, m))
        match_sq = match * match

        for i in range(order - 1):
            aux1 = S * Kp[i]
            aux2 = aux1.dot(D[0:m, 0:m])
            Kpp = match_sq * aux2
            Kp[i + 1] = Kpp.T.dot(D[0:n, 0:n]).T

            daux1_dgap = S * dKp_dgap[i]
            daux2_dgap = daux1_dgap.dot(D[0:m, 0:m]) + aux1.dot(dD_dgap[0:m, 0:m])
            dKpp_dgap = match_sq * daux2_dgap
            dKp_dgap[i + 1] = dKpp_dgap.T.dot(D[0:n, 0:n]).T + Kpp.T.dot(dD_dgap[0:n, 0:n]).T

            daux1_dmatch = S * dKp_dmatch[i]
            daux2_dmatch = daux1_dmatch.dot(D[0:m, 0:m])
            dKpp_dmatch = (match_sq * daux2_dmatch) + (2 * match * aux2)
            dKp_dmatch[i + 1] = dKpp_dmatch.T.dot(D[0:n, 0:n]).T

        # Final calculation
        aux1 = S * Kp
        aux2 = np.sum(aux1, axis=1)
        aux3 = np.sum(aux2, axis=1)
        Ki = match_sq * aux3
        k = Ki.dot(coefs)

        daux1_dgap = S * dKp_dgap
        daux2_dgap = np.sum(daux1_dgap, axis=1)
        daux3_dgap = np.sum(daux2_dgap, axis=1)
        dKi_dgap = match_sq * daux3_dgap
        dk_dgap = dKi_dgap.dot(coefs)
        
        daux1_dmatch = S * dKp_dmatch
        daux2_dmatch = np.sum(daux1_dmatch, axis=1)
        daux3_dmatch = np.sum(daux2_dmatch, axis=1)
        dKi_dmatch = match_sq * daux3_dmatch + (2 * match * aux3)
        dk_dmatch = dKi_dmatch.dot(coefs)

        dk_dcoefs = Ki

        return k, dk_dgap, dk_dmatch, dk_dcoefs

    def _dot(self, s1, s2):
        """
        Simple dot product between two vectors of embeddings.
        This returns a matrix of positive real numbers.
        """
        embs1 = self.embs[s1]
        embs2 = self.embs[s2]
        return embs1.dot(embs2.T)

    def _arccosine(self, s1, s2):
        """
        Uses an arccosine kernel of degree 0 to calculate
        the similarity matrix between two vectors of embeddings. 
        This is just cosine similarity projected into the [0,1] interval.
        """
        embs1 = self.embs[s1]
        embs2 = self.embs[s2]
        normembs1 = self.norms[s1]
        normembs2 = self.norms[s2]
        norms = np.dot(normembs1, normembs2.T)
        dot = embs1.dot(embs2.T)
        # We clip values due to numerical errors
        # which put some values outside the arccosine range.
        cosine = np.clip(dot / norms, -1, 1)
        angle = np.arccos(cosine)
        return 1 - (angle / np.pi)

    def K(self, X, X2, gram, params, diag=False):
        """
        Calculates and returns the Gram matrix between two lists
        of strings. These should be encoded as lists of integers.
        """
        order = len(params[2])
        if diag:
            # Assume only X is given
            k_result = np.zeros(shape=(len(X)))
            gap_grads = np.zeros(shape=(len(X)))
            match_grads = np.zeros(shape=(len(X)))
            coef_grads = np.zeros(shape=(len(X), order))
            for i, x1 in enumerate(X):
                result = self._k(x1[0], x1[0], params)
                k_result[i] = result[0]
                gap_grads[i] = result[1]
                match_grads[i] = result[2]
                coef_grads[i] = np.array(result[3:])
        else:
            k_result = np.zeros(shape=(len(X), len(X2)))
            gap_grads = np.zeros(shape=(len(X),len(X2)))
            match_grads = np.zeros(shape=(len(X), len(X2)))
            coef_grads = np.zeros(shape=(len(X), len(X2), order))
            for i, x1 in enumerate(X):
                for j, x2 in enumerate(X2):
                    if gram and (j < i):
                        k_result[i, j] = k_result[j, i]
                        gap_grads[i, j] = gap_grads[j, i]
                        match_grads[i, j] = match_grads[j, i]
                        coef_grads[i, j] = coef_grads[j, i]
                    else:
                        result = self._k(x1[0], x2[0], params)
                        k_result[i, j] = result[0]
                        gap_grads[i, j] = result[1]
                        match_grads[i, j] = result[2]
                        coef_grads[i, j] = np.array(result[3:])
        return k_result, gap_grads, match_grads, coef_grads

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


def encode_string(s, index):
    """
    Transform a string in a list of integers.
    The ints correspond to indices in an
    embeddings matrix.
    """
    return [index[symbol] for symbol in s]
