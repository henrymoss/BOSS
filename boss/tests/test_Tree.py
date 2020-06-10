# # import external packages
# import unittest
# import numpy as np
# import GPy
# import sys

# # import our code
# from boss.code.GPy_wrappers.GPy_tree_kernel import SubsetTreeKernel
# from boss.code.kernels.tree.C_tree_kernel import wrapper_raw_SubsetTreeKernel

# # test tree kernel calculations

# class STKTests(unittest.TestCase):
#     """
#     Tests for the tree kernel
#     """

#     # test of raw C code (on example from Beck 2015)

#     def test_C(self):
#         X = np.array([["(C (B c) (B a))"]],dtype=object)
#         output=[]
#         # check 1: count each fragment = 6
#         STK=wrapper_raw_SubsetTreeKernel(_lambda=1,_sigma=1,normalize=False)
#         output.append(STK.K(X,X)[0][0][0])
        
#         # check 2: count fragments with ending only terminal symbols = 3
#         STK=wrapper_raw_SubsetTreeKernel(_lambda=1,_sigma=0,normalize=False)
#         output.append(STK.K(X,X)[0][0][0])
        
#         # check 3: weight fragments of length i  by 0.5^i 
#         STK=wrapper_raw_SubsetTreeKernel(_lambda=0.5,_sigma=1,normalize=False)
#         # should be 2*lambda + lamba(alpha+lambda)^2
#         output.append(STK.K(X,X)[0][0][0])
#         self.assertEqual(output, [6,3,2.125])


#     #tests for single kernel calculations across different hyper-parameters


#     def test_single1(self):
#         tk = SubsetTreeKernel(normalize=False)
#         tk._set_params([1,0])
#         t = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
#         X1 = np.array([[t]], dtype=object)
#         target = tk.K(X1, X1)
#         self.assertEqual(target[0], [7])

#     def test_single2(self):
#         tk = SubsetTreeKernel(normalize=False)
#         tk._set_params([1,1])
#         t = "(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))"
#         X1 = np.array([[t]], dtype=object)
#         target = tk.K(X1, X1)
#         self.assertEqual(target[0], [37])

#     def test_single3(self):
#         tk = SubsetTreeKernel(normalize = False)
#         X = np.array([['(S (NP a) (VP v))']], dtype=object)
#         X2 = np.array([['(S (NP (NP a)) (VP (V c)))']], dtype=object)
#         k = tk.K(X, X2)
#         self.assertTrue((k == 2))

#     # tests for multiple kernel calculations (normalized and not normalized)

#     def test_multiple_diag_normed(self):
#         tk = SubsetTreeKernel()
#         X = np.array([['(S (NP ns) (VP v))'],
#                       ['(S (NP n) (VP v))'],
#                       ['(S (NP (N a)) (VP (V c)))'],
#                       ['(S (NP (Det a) (N b)) (VP (V c)))'],
#                       ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
#                      dtype=object)
#         diag = tk.Kdiag(X)
#         self.assertTrue(([1,1,1,1,1] == diag).all())

#     def test_multiple_full_normed(self):
#         X = np.array([['(S (NP ns) (VP v))'],
#                       ['(S (NP n) (VP v))'],
#                       ['(S (NP (N a)) (VP (V c)))'],
#                       ['(S (NP (Det a) (N b)) (VP (V c)))'],
#                       ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
#                      dtype=object)
#         k = SubsetTreeKernel()
#         output = k.K(X, None)
#         result = [[ 1.,          0.5,         0.10540926,  0.08333333,  0.06711561],
#                   [ 0.5,         1.,          0.10540926,  0.08333333,  0.06711561],
#                   [ 0.10540926,  0.10540926,  1.,          0.31622777,  0.04244764],
#                   [ 0.08333333,  0.08333333,  0.31622777,  1.,          0.0335578 ],
#                   [ 0.06711561,  0.06711561,  0.04244764,  0.0335578,   1.        ]]
#         self.assertAlmostEqual(np.sum(result), np.sum(output))

#     def test_multiple_diag_no_norm(self):
#         tk = SubsetTreeKernel(normalize = False)
#         X = np.array([['(S (NP ns) (VP v))'],
#                       ['(S (NP n) (VP v))'],
#                       ['(S (NP (N a)) (VP (V c)))'],
#                       ['(S (NP (Det a) (N b)) (VP (V c)))'],
#                       ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
#                      dtype=object)
#         diag = tk.Kdiag(X)
#         self.assertTrue(([6,6,15,24,37] == diag).all())

#     def test_multiple_full_no_norm(self):
#         tk = SubsetTreeKernel(normalize=False)
#         X = np.array([['(S (NP a) (VP v))'],
#                       ['(S (NP a1) (VP v))'],
#                       ['(S (NP (NP a)) (VP (V c)))'],
#                       ['(S (VP v2))']],
#                      dtype=object)
#         k = tk.K(X, X)
#         result = np.array([[6,3,2,0],
#                            [3,6,1,0],
#                            [2,1,15,0],
#                            [0,0,0,3]])
#         self.assertTrue((k == result).all())


#     #tests for gradients 


#     def test_kernel_grad_no_norm(self):
#         # compare with exact
#         tk = SubsetTreeKernel(normalize = False)
#         X = np.array([['(S (NP a) (VP v))']], dtype=object)
#         X2 = np.array([['(S (NP (NP a)) (VP (V c)))']], dtype=object)
#         k = tk.dK_dtheta(1, X, X2)
#         self.assertTrue((k == [2,2]).all())




#     def test_kernel_grad_norm(self):
#         # compare with approx
#         tk = SubsetTreeKernel(_lambda=1, _sigma=1)
#         X = np.array([['(S (NP ns) (VP v))'],
#                       ['(S (NP n) (VP v))'],
#                       ['(S (NP (N a)) (VP (V c)))'],
#                       ['(S (NP (Det a) (N b)) (VP (V c)))'],
#                       ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
#                      dtype=object)
#         h = 0.00001
#         k = tk.K(X)
#         dk_dt = tk.dK_dtheta(1, X, None)
#         tk[''] = [1,1-h]
#         k_b1 = tk.K(X)
#         tk[''] = [1,1+h]
#         k_b2 = tk.K(X)
#         tk[''] = [1-h,1]
#         k_d1 = tk.K(X)
#         tk[''] = [1+h,1]
#         k_d2 = tk.K(X)
#         approx = [np.sum((k_d2 - k_d1) / (2 * h)), np.sum((k_b2 - k_b1) / (2 * h))]
#         self.assertAlmostEqual(approx[0], dk_dt[0])
#         self.assertAlmostEqual(approx[1], dk_dt[1])


#     def test_kernel_grad_norm_different_params(self):
#         L = 0.5
#         S = 5
#         tk = SubsetTreeKernel( _lambda=L, _sigma=S)
#         X = np.array([['(S (NP a) (VP v))']], dtype=object)
#         X2 = np.array([['(S (NP (NP a)) (VP (V c)))']], dtype=object)
#         h = 0.00001
#         dk_dt = tk.dK_dtheta(1, X, X2)
#         tk._set_params([L,S-h])
#         k_b1 = tk.K(X, X2)
#         tk._set_params([L,S+h])
#         k_b2 = tk.K(X, X2)
#         tk._set_params([L-h,S])
#         k_d1 = tk.K(X, X2)
#         tk._set_params([L+h,S])
#         k_d2 = tk.K(X, X2)
#         tk._set_params([1,1])
#         approx = [np.sum((k_d2 - k_d1) / (2 * h)), np.sum((k_b2 - k_b1) / (2 * h))]
#         self.assertAlmostEqual(approx[0], dk_dt[0])
#         self.assertAlmostEqual(approx[1], dk_dt[1])






# # class IntegrationSTKTests(unittest.TestCase):
# #     """
# #     The goal of these tests is to check if "something" is done when used in GPy. =P
# #     There are no asserts here, so only an Error is considered a Failure.
# #     """

# #     def test_mode_fit(self):
# #         tk = SubsetTreeKernel()
# #         X = np.array([['(S NP VP)'], ['(S NP ADJ)'], ['(S NP)']], dtype=object)
# #         Y = np.array([[1],[2],[3]])
# #         m = GPy.models.GPRegression(X, Y, kernel=tk)

# #     def test_model_optimize(self):
# #         tk = SubsetTreeKernel()
# #         X = np.array([['(S (NP N) (VP V))'], ['(S NP ADJ)'], ['(S NP)']], dtype=object)
# #         Y = np.array([[1],[2],[30]])
# #         m = GPy.models.GPRegression(X, Y, kernel=tk)
# #         m.optimize(max_f_eval=50)
    

# #     def test_model_optimize_larger(self):
# #         tk = SubsetTreeKernel()
# #         X = np.array([['(S NP VP)'],
# #                       ['(S (NP N) (VP V))'],
# #                       ['(S (NP (N a)) (VP (V c)))'],
# #                       ['(S (NP (Det a) (N b)) (VP (V c)))'],
# #                       ['(S (NP (ADJ colorless) (N ideas)) (VP (V sleep) (ADV furiously)))']],
# #                      dtype=object)
# #         Y = np.array([[(a+10)*5] for a in range(5)])
# #         m = GPy.models.GPRegression(X, Y, kernel=tk)
# #         m.optimize(max_f_eval=10)







# if __name__ == '__main__':
#     print("Running unit tests")
#     unittest.main()
