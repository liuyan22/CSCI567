import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################
		h = sum([self.clfs_picked[i].predict(features)*self.betas[i] for i in range(self.T)])
		H = np.sign(np.array(h))
		return H.tolist()
		




class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################
		N = len(features)
		w = np.ones(N)/N
		for t in range(self.T):
			indicator_func = []
			for i in range(len(list(features))):
				indicator_func_var = (labels != list(self.clfs)[i].predict(features))
				indicator_func.append(indicator_func_var)
			h_t = np.argmin(np.dot(w, indicator_func), axis=0)
			e_t = np.dot(w, np.not_equal(labels, list(self.clfs)[h_t].predict(features)))
			beta_t = 0.5*np.log(1-e_t)/e_t
			indicator_func_2 = [x if x==1 else -1 for x in indicator_func]
			w = np.multiply(w, np.exp([float(x)*beta_t for x in indicator_func_2]))
			w = w/np.sum(w)

		


		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	