# import external packages
import unittest
import numpy as np
import GPy
import sys
from copy import deepcopy

# import our code
sys.path.append('../')
from Kernels.String.GPy_string_kernel import StringKernel
from Legacy.old_string_kernel import OLDStringKernel


#Checks for our tensorflow implementation of the string kernel (good for large strings on GPU)
# outperformed by numpy on CPU and for many smaller strings

class Kern_Cal_Tests(unittest.TestCase):
	"""
	Test the kernel calculations on some simple hardcoded examples
	Checking that shortcuts (e.g diag and sym) work properly
	"""
	def setUp(self):
		self.s1 = 'c a t a'
		self.s2 = 'g a t t a'
		self.s3 = 'c g t a g c t a g c g a c g c a g c c a a t c g a t c g'
		self.s4 = 'c g a g a t g c c a a t a g a g a g a g c g c t g t a'
		self.X = np.array([[self.s1],[self.s2],[self.s3],[self.s4]])
		self.alphabet = 'acgt'
		self.maxlen=28



	def test_k_1(self):
		self.kern = StringKernel(implementation="tensorflow", alphabet=self.alphabet,order_coefs = [1.] * 5,normalize=False,
									gap_decay = 2.0, match_decay = 2.0,maxlen=self.maxlen)
		expected = 504.0
		result = self.kern.K(self.X,self.X)
		self.assertAlmostEqual(result[0][1], expected)


	def test_k_2(self):
		self.kern = StringKernel(implementation="tensorflow", alphabet=self.alphabet,order_coefs = [1.] * 5,normalize=False,
									gap_decay = 0.8, match_decay = 0.8, maxlen=self.maxlen)
		expected = 5.943705
		result = self.kern.K(self.X,self.X)
		self.assertAlmostEqual(result[0][1], expected, places=4)



	def test_diag_non_diag(self):
		self.kern = StringKernel(implementation="tensorflow", alphabet=self.alphabet,order_coefs = [1.] * 5,normalize=False,
									gap_decay = 2.0, match_decay = 2.0,maxlen=self.maxlen)
		result1 = np.diag(self.kern.K(self.X))
		result2 = self.kern.Kdiag(self.X)
		self.assertAlmostEqual(np.sum(result1)/1000, np.sum(result2)/1000)


	def test_sym_non_sym(self):
		self.kern = StringKernel(implementation="tensorflow", alphabet=self.alphabet,order_coefs = [1.] * 5,normalize=False,
									gap_decay = 2.0, match_decay = 2.0,maxlen=self.maxlen)
		result1 = self.kern.K(self.X)
		result2 = self.kern.K(self.X, self.X)
		self.assertAlmostEqual(np.sum(result1)/1000, np.sum(result2)/1000)


	def test_norm_diag(self):
		self.kern = StringKernel(implementation="tensorflow", alphabet=self.alphabet,order_coefs = [1.] * 5,normalize=True,
									gap_decay = 2.0, match_decay = 2.0,maxlen=self.maxlen)
		result = self.kern.Kdiag(self.X)
		self.assertTrue(np.array_equal(result,np.ones((len(self.X)))))

	def test_norm_no_diag(self):
		self.kern = StringKernel(implementation="tensorflow", alphabet=self.alphabet,order_coefs = [1.] * 5,normalize=True,
									gap_decay = 2.0, match_decay = 2.0,maxlen=self.maxlen)
		result = np.diag(self.kern.K(self.X))
		self.assertTrue(np.array_equal(result,np.ones((len(self.X)))))

class Legacy_Tests(unittest.TestCase):
	"""
	Check that our implementation matches the output from our old pure python implementation
	Also check gradients
	"""
	def setUp(self):
		self.s1 = 'c a t a'
		self.s2 = 'g a t t a'
		self.s3 = 'c g t a g c t a g c g a c g c a g c c a a t c g a t c g'
		self.s4 = 'c g a g a t g c c a a t a g a g a g a g c g c t g t a'
		self.X = np.array([[self.s1],[self.s2],[self.s3],[self.s4]])
		self.s1_old = 'cata'
		self.s2_old = 'gatta'
		self.s3_old = 'cgtagctagcgacgcagccaatcgatcg'
		self.s4_old = 'cgagatgccaatagagagagcgctgta'
		self.X_old = np.array([[self.s1_old],[self.s2_old],[self.s3_old],[self.s4_old]])
		self.alphabet = ['a','c','g','t']
		self.maxlen=28

	def test_old_1(self):
		# check old code still works
		self.kern = OLDStringKernel(alphabet=self.alphabet,order_coefs = [1.] * 5,
									gap_decay = 2.0, match_decay = 2.0)
		expected = 504.0
		result = self.kern.K(self.X_old,self.X_old)
		self.assertAlmostEqual(result[0][1], expected)

	def test_old_2(self):
		# check old code still works
		self.kern = OLDStringKernel(alphabet=self.alphabet,order_coefs = [1.] * 5,
									gap_decay = 0.8, match_decay = 0.8)
		expected = 5.943705
		result = self.kern.K(self.X_old,self.X_old)
		self.assertAlmostEqual(result[0][1], expected, places=4)


	def test_compare_grads_1(self):
		self.kern_new = StringKernel(implementation="tensorflow", alphabet=self.alphabet,order_coefs = [0.1, 0.2, 0.4, 0.5] ,
									gap_decay = 0.8, match_decay = 0.8,maxlen=self.maxlen,normalize=False)
		self.kern_old = OLDStringKernel( alphabet=self.alphabet,order_coefs = [0.1, 0.2, 0.4, 0.5] ,
									gap_decay = 0.8, match_decay = 0.8)

		result1 = self.kern_new.K(self.X)
		result2 = self.kern_old.K(self.X_old)
		np.set_printoptions(suppress=True)

		self.assertAlmostEqual(np.sum(result1), np.sum(result2), places=2)
		self.assertAlmostEqual(np.sum(self.kern_new.gap_grads)/1000, np.sum(self.kern_old.gap_grads)/1000, places=2)
		self.assertAlmostEqual(np.sum(self.kern_new.match_grads)/1000, np.sum(self.kern_old.match_grads)/1000, places=2)
		self.assertAlmostEqual(np.sum(self.kern_new.coef_grads)/1000, np.sum(self.kern_old.coef_grads)/1000, places=2)


	def test_compare_grads_2(self):
		self.kern_new = StringKernel(implementation="tensorflow", alphabet=self.alphabet,order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7] ,
									gap_decay = 2, match_decay = 2, maxlen=self.maxlen,normalize=False)
		self.kern_old = OLDStringKernel(alphabet=self.alphabet,order_coefs = [0.1, 0.2, 0.4, 0.5, 0.7] ,
									gap_decay = 2, match_decay = 2)

		result1 = self.kern_new.K(self.X)
		result2 = self.kern_old.K(self.X_old)
		np.set_printoptions(suppress=True)

		self.assertAlmostEqual(np.sum(result1), np.sum(result2), places=2)
		self.assertAlmostEqual(np.sum(self.kern_new.gap_grads)/1000, np.sum(self.kern_old.gap_grads)/1000, places=2)
		self.assertAlmostEqual(np.sum(self.kern_new.match_grads)/1000, np.sum(self.kern_old.match_grads)/1000, places=2)
		self.assertAlmostEqual(np.sum(self.kern_new.coef_grads)/1000, np.sum(self.kern_old.coef_grads)/1000, places=2)

class Gradient_Tests(unittest.TestCase):
	"""
	Check that our gradients match empirical gradients
	"""
	def setUp(self):
		self.s1 = 'c a t a'
		self.s2 = 'g a t t a'
		self.X1 = np.array([[self.s1]])
		self.X2 = np.array([[self.s2]])
		self.alphabet = ['a','c','g','t']
		self.maxlen=28


	def test_gradient_gap_no_norm(self):

		self.kern = StringKernel(implementation="tensorflow", alphabet=self.alphabet,order_coefs = [0.1,0.1],normalize=False,
									gap_decay = 0.8, match_decay = 0.8, maxlen=self.maxlen)
		result = self.kern.K(self.X1, self.X2)
		true_grads = self.kern.gap_grads

		E = 1e-4
		self.kern.gap_decay.constrain_fixed( 0.8 + E )
		g_result1 = self.kern.K(self.X1, self.X2)
		self.kern.gap_decay.constrain_fixed(0.8 - E)
		g_result2 = self.kern.K(self.X1, self.X2)
		g_result = (g_result1 - g_result2) / (2 * E)
		self.assertAlmostEqual(true_grads[0][0], g_result[0][0], places=2)		

	def test_gradient_gap_norm(self):

		self.kern = StringKernel(implementation="tensorflow", alphabet=self.alphabet,order_coefs = [0.1,0.1],normalize=True,
									gap_decay = 0.8, match_decay = 0.8, maxlen=self.maxlen)
		result = self.kern.K(self.X1, self.X2)
		true_grads = self.kern.gap_grads

		E = 1e-4
		self.kern.gap_decay.constrain_fixed( 0.8 + E )
		g_result1 = self.kern.K(self.X1, self.X2)
		self.kern.gap_decay.constrain_fixed(0.8 - E)
		g_result2 = self.kern.K(self.X1, self.X2)
		g_result = (g_result1 - g_result2) / (2 * E)
		self.assertAlmostEqual(true_grads[0][0], g_result[0][0], places=2)		


	def test_gradient_coefs_dim_1_no_norm(self):
		self.kern = StringKernel(implementation="tensorflow", alphabet=self.alphabet,order_coefs = [0.1,0.1],normalize=False,
									gap_decay = 0.8, match_decay = 0.8, maxlen=self.maxlen)
		result = self.kern.K(self.X1, self.X2)
		true_grads = self.kern.coef_grads

		E = 1e-4
		self.kern.order_coefs.constrain_fixed([0.1 + E, 0.1])
		g_result1 = self.kern.K(self.X1, self.X2)
		self.kern.order_coefs.constrain_fixed([0.1 - E, 0.1])
		g_result2 = self.kern.K(self.X1, self.X2)
		g_result = (g_result1 - g_result2) / (2 * E)
		self.assertAlmostEqual(true_grads[0][0][0], g_result[0][0], places=2)

	def test_gradient_coefs_dim_2_no_norm(self):
		self.kern = StringKernel(implementation="tensorflow", alphabet=self.alphabet,order_coefs = [0.1,0.1],normalize=False,
									gap_decay = 0.8, match_decay = 0.8, maxlen=self.maxlen)
		result = self.kern.K(self.X1, self.X2)
		true_grads = self.kern.coef_grads

		E = 1e-4
		self.kern.order_coefs.constrain_fixed([0.1 , 0.1 + E])
		g_result1 = self.kern.K(self.X1, self.X2)
		self.kern.order_coefs.constrain_fixed([0.1, 0.1 - E])
		g_result2 = self.kern.K(self.X1, self.X2)
		g_result = (g_result1 - g_result2) / (2 * E)
		self.assertAlmostEqual(true_grads[0][0][1], g_result[0][0], places=2)

	def test_gradient_coefs_dim_1_norm(self):
		self.kern = StringKernel(implementation="tensorflow", alphabet=self.alphabet,order_coefs = [0.1,0.1],normalize=True,
									gap_decay = 0.8, match_decay = 0.8, maxlen=self.maxlen)
		result = self.kern.K(self.X1, self.X2)
		true_grads = self.kern.coef_grads

		E = 1e-4
		self.kern.order_coefs.constrain_fixed([0.1 + E, 0.1])
		g_result1 = self.kern.K(self.X1, self.X2)
		self.kern.order_coefs.constrain_fixed([0.1 - E, 0.1])
		g_result2 = self.kern.K(self.X1, self.X2)
		g_result = (g_result1 - g_result2) / (2 * E)
		self.assertAlmostEqual(true_grads[0][0][0], g_result[0][0], places=2)

	def test_gradient_coefs_dim_2_norm(self):
		self.kern = StringKernel(implementation="tensorflow", alphabet=self.alphabet,order_coefs = [0.1,0.1],normalize=True,
									gap_decay = 0.8, match_decay = 0.8, maxlen=self.maxlen)
		result = self.kern.K(self.X1, self.X2)
		true_grads = self.kern.coef_grads

		E = 1e-4
		self.kern.order_coefs.constrain_fixed([0.1 , 0.1 + E])
		g_result1 = self.kern.K(self.X1, self.X2)
		self.kern.order_coefs.constrain_fixed([0.1, 0.1 - E])
		g_result2 = self.kern.K(self.X1, self.X2)
		g_result = (g_result1 - g_result2) / (2 * E)
		self.assertAlmostEqual(true_grads[0][0][1], g_result[0][0], places=2)



	def test_gradient_match_no_norm(self):

		self.kern = StringKernel(implementation="tensorflow", alphabet=self.alphabet,order_coefs = [0.1,0.1],normalize=False,
									gap_decay = 0.8, match_decay = 0.8, maxlen=self.maxlen)
		result = self.kern.K(self.X1, self.X2)
		true_grads = self.kern.match_grads

		E = 1e-4
		self.kern.match_decay.constrain_fixed( 0.8 + E )
		g_result1 = self.kern.K(self.X1, self.X2)
		self.kern.match_decay.constrain_fixed(0.8 - E)
		g_result2 = self.kern.K(self.X1, self.X2)
		g_result = (g_result1 - g_result2) / (2 * E)
		self.assertAlmostEqual(true_grads[0][0], g_result[0][0], places=2)		

	def test_gradient_match_norm(self):

		self.kern = StringKernel(implementation="tensorflow", alphabet=self.alphabet,order_coefs = [0.1,0.1],normalize=True,
									gap_decay = 0.8, match_decay = 0.8, maxlen=self.maxlen)
		result = self.kern.K(self.X1, self.X2)
		true_grads = self.kern.match_grads

		E = 1e-4
		self.kern.match_decay.constrain_fixed( 0.8 + E )
		g_result1 = self.kern.K(self.X1, self.X2)
		self.kern.match_decay.constrain_fixed(0.8 - E)
		g_result2 = self.kern.K(self.X1, self.X2)
		g_result = (g_result1 - g_result2) / (2 * E)
		self.assertAlmostEqual(true_grads[0][0], g_result[0][0], places=2)		




class IntegrationTests(unittest.TestCase):
	"""
	The goal of these tests is to check if "something" is done when used in GPy.
	There are no asserts here, so only an Error is considered a Failure.
	"""
	def setUp(self):
		self.s1 = 'c a t a'
		self.s2 = 'g a t t a'
		self.s3 = 'c g t a g c t a g c g a c g c a g c c a a t c g a t c g'
		self.s4 = 'c g a g a t g c c a a t a g a g a g a g c g c t g t a'
		self.X = np.array([[self.s1],[self.s2],[self.s3],[self.s4]])
		self.alphabet = ['a','c','g','t']
		self.maxlen=28

	def test_mode_fit(self):
		self.kern = StringKernel(implementation="tensorflow", alphabet=self.alphabet,order_coefs = [0.1,0.1],normalize=True,
									gap_decay = 0.8, match_decay = 0.8, maxlen=self.maxlen)
		Y = np.array([[1],[2],[3],[4]])
		m = GPy.models.GPRegression(self.X, Y, kernel=self.kern)

	def test_model_optimize(self):
		self.kern = StringKernel(implementation="tensorflow", alphabet=self.alphabet,order_coefs = [0.1,0.1],normalize=True,
									gap_decay = 0.8, match_decay = 0.8, maxlen=self.maxlen)
		Y = np.array([[1],[2],[3],[4]])
		m = GPy.models.GPRegression(self.X, Y, kernel=self.kern)
		m.optimize(max_f_eval=50)
	

if __name__ == '__main__':
	print("Running unit tests")
	unittest.main()

