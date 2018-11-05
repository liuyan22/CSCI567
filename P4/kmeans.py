import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (means a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
            Note: Number of iterations is the number of time you update the assignment
        ''' 
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
         #   'Implement fit function in KMeans class (filename: kmeans.py)')
        J = 10^10
        init_means = np.random.choice(N, self.n_cluster, replace=True)
        means = x[init_means]
        distances = np.zeros((N, self.n_cluster))
        membership = np.zeros((N))
        r_ik = np.zeros((N, self.n_cluster))
        for i in range(self.max_iter):
            number_of_updates = i
            for j in range(self.n_cluster):
                distances[:, j] = np.sum(np.square(means[j] - x), axis=1)
            membership = np.argmin(distances, axis=1)
            r_ik = np.identity(self.n_cluster)[membership]
            Jnew = 0
            #print(r_ik)
            for i in range(N):
                means_i = means[np.argmax(r_ik[i])]
                diff = means_i - x[i]
                Jnew += np.dot(diff.T,diff)
            Jnew = Jnew/N
            #print(Jnew)
            if abs(J-Jnew) <= self.e:
                break
            J = Jnew
            for k in range(self.n_cluster):
                x_k = x[membership == k]
                means[k] = np.apply_along_axis(np.mean, axis=0, arr=x_k)
        return means, membership, number_of_updates
        # DONOT CHANGE CODE BELOW THIS LINE

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float) 
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.means : means obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by 
                    majority voting ((N,) numpy array) 
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to means
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
        #    'Implement fit function in KMeansClassifier class (filename: kmeans.py)')
        kmeans_clf = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)
        means, membership, number_of_updates = kmeans_clf.fit(x)
        centroid_labels = np.zeros((self.n_cluster))
        #print(y.shape)
        for i in range(self.n_cluster):
            index = np.where(membership == i)
            #print(index)
            votes = np.bincount(y[index])
            #print(votes)
            centroid_labels[i] = np.argmax(votes)
            #print(centroid_labels)
        centroids = means
        self.centroids = centroids
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.means = means

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(
            self.n_cluster)

        assert self.means.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
        #    'Implement predict function in KMeansClassifier class (filename: kmeans.py)')
        distances = np.zeros((N, self.n_cluster))
        membership = np.zeros((N))
        labels = np.zeros((N))
        r_ik = np.zeros((N, self.n_cluster))
        for j in range(self.n_cluster):
            distances[:, j] = np.sum(np.square(self.means[j] - x), axis=1)
        membership = np.argmin(distances, axis=1)
        for i in range(N):
            r_ik[i][membership[i]] = 1
            labels[i] = self.centroid_labels[np.argmax(r_ik[i])]
        # DONOT CHANGE CODE BELOW THIS LINE
        return labels

