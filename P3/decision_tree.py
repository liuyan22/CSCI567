import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		
		string = ''
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples 
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●
				               
				      branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			branches = np.array(branches)
			num_elements = np.sum(branches)
			entropy = 0
			cond_entropy = 0
			for i in range(branches.shape[1]):
				branch = branches[:,i]/np.sum(branches[:,i])
				branch[np.where(branch == 0)[0]] += 1
				probability = np.sum(branches[:,i])/num_elements
				for j in range(len(branch)):
					if np.log2(branch[j]) != 0:
						entropy -=  -np.log2(branch[j])*branch[j]
				cond_entropy -= cond_entropy + entropy*probability
			return cond_entropy
			

		features = np.array(self.features)
		entropies = []
		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		############################################################
			feature = features[:,idx_dim]
			branches = []
			for m in np.unique(feature):
				m_labels = np.array(self.labels)[np.where(feature==m)]
				branch = []
				for i in range(self.num_cls):
					branch.append(np.sum(m_labels==i))
				branches.append(branch)
			entropies.append(conditional_entropy(np.array(branches).T.tolist()))
		self.dim_split = np.argmin(entropies)
		
		############################################################
		# TODO: split the node, add child nodes
		############################################################
		feature = features[:,self.dim_split]
		self.feature_uniq_split = np.unique(feature).tolist()
		if len(np.unique(feature)) > 1:
			for m in np.unique(feature):
				self.children.append(TreeNode(features[np.where(feature==m)].tolist(),np.array(self.labels)[np.where(feature==m)].tolist(), self.num_cls))
		else:
			self.splittable = False

		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return




		
	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max





