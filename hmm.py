import numpy as np
from scipy import stats
from numpy import random




class GaussianHMM:
    def __init__(self, num_hidden_states=1, dim_observation=1, init_parameters=True):
        self.__num_hidden_states = num_hidden_states
        self.__dim_observation = dim_observation
        self.__has_parameters = False
        if init_parameters:
            # init start prob, num_hidden_states X 1
            tmp = random.random([self.__num_hidden_states, 1])
            tmp = tmp/(np.sum(tmp))
            self.__start_probability = tmp
            # init transition prob, num_hidden_states x num_hidden_states
            tmp = random.random([self.__num_hidden_states, self.__num_hidden_states])
            row_sum = np.sum(tmp, axis=1)
            self.__transition_probability = (tmp.T/row_sum).T
            # init emission prob
            mu = random.random([self.__dim_observation, self.__num_hidden_states])-1
            sigma = random.random([self.__num_hidden_states, self.__dim_observation, self.__dim_observation])+0.5
            self.__emission = [mu, sigma]
            self.__has_parameters = True

    def def_parameters(self, start, transition, emission):
        assert self.__num_hidden_states==start.shape[0]==transition.shape[0]==transition.shape[1]==emission[0].shape[1]==emission[1].shape[0], 'num_hidden_states are not consistent!'
        assert self.__dim_observation==emission[0].shape[0]==emission[1].shape[1]==emission[1].shape[2], 'dim_observation are not consistent'
        self.__start_probability = start, self.__transition_probability = transition, self.__emission = emission
        self.__has_parameters = True

    def train(self, X, max_iter=100, tol=0.01, print_interval=10):
        self.__P_X = []
        for i in range(max_iter):
            P_X = self.__EM(X, tol)
            self.__P_X.append(P_X)
            if i%print_interval == 0:
                print('iter:', i, 'P_X:', P_X)
        print('done. num of iteration has reached max_iter. P_X:', self.__P_X[-1])

    def __EM(self, X, tol):
        # E step
        self.__c = np.zeros(X.shape[-1])
        self.__alpha = np.zeros([self.__num_hidden_states, X.shape[-1]]) # num_hidden_states X num_samples
        self.__beta = np.zeros_like(self.__alpha) # num_hidden_states X num_samples
        self.__cal_alpha(X, X.shape[-1])
        self.__cal_beta(X, 1)
        self.__gamma = self.__alpha * self.__beta
        self.__xi = np.zeros([X.shape[-1]-1, self.__num_hidden_states, self.__num_hidden_states])
        for i in range(X.shape[-1]-1):
            self.__xi[i, :, :] = self.__c[i + 1] * np.dot(np.reshape(self.__alpha[:, i], [-1, 1]), self.__emission_dist(X[:, i+1]).T * self.__beta[:, i+1].T) * self.__transition_probability
        P_X = self.__c.prod()
        # M step

        pass

    def __cal_alpha(self, X, n):
        if(n > 1):
            self.__cal_alpha(X, n-1)
            self.__c[n-1] = np.dot(np.dot(self.__alpha[:, n-2].T, self.__transition_probability), self.__emission_dist(X[:, n-1]))
            self.__alpha[:, n-1] = self.__emission_dist(X[:, n-1]) * np.dot(self.__alpha[:, n-2].T, self.__transition_probability).T / self.__c[n-1]
            return
        self.__c[n-1] = np.dot(self.__start_probability.T, self.__emission_dist(X[:, n-1]))
        self.__alpha[:, n-1] = self.__start_probability * self.__emission_dist(X[:, n-1]) / self.__c[n-1]
        # return self.__alpha[:, n-1]
        return


    def __cal_beta(self, X, n):
        if(n < X.shape[-1]):
            pass
        self.__beta[] =

    def __emission_dist(self, x):
        prob = np.zeros([self.__num_hidden_states, 1])
        for i in range(self.__num_hidden_states):
            prob[i, 0] = stats.multivariate_normal.pdf(x, self.__emission[0][:, i], self.__emission[1][i, :, :])
        return prob


if __name__ == '__main__':
    model = GaussianHMM(2)
