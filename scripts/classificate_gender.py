#!/usr/bin/python

import numpy as np
import os
import sys
import pandas as pd
from scipy.spatial.distance import *
from scipy.sparse.csgraph import dijkstra, shortest_path, connected_components, laplacian

from sklearn.base import  BaseEstimator, TransformerMixin
from copy import deepcopy

from sklearn.model_selection import GroupKFold, cross_val_score, GroupShuffleSplit
import networkx as nx
import igraph as ig
import scipy
import time

from reskit.core import Pipeliner

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit


#########
# Norms #
#########


def no_norm(matrix):
    return matrix

def max_norm(matrix):
    normed_matrix = matrix / np.max(matrix)
    return normed_matrix

def binar_norm(matrix):
    bin_matrix = matrix.copy()
    bin_matrix[bin_matrix > 0] = 1
    return bin_matrix

def mean_norm(matrix):
    normed_matrix = matrix / np.mean(matrix)
    return normed_matrix

def double_norm(function, matrix1, matrix2):
    return function(matrix1), function(matrix2)


###############
# Featurizers #
###############


def bag_of_edges(X, SPL=None, symmetric = True, return_df = False, offset = 1):
    size = X.shape[1]
    if symmetric:
        indices = np.triu_indices(size, k = offset)
    else:
        grid = np.indices(X.shape[1:])
        indices = (grid[0].reshape(-1), grid[1].reshape(-1))
    if len(X.shape) == 3:
        featurized_X = X[:, indices[0], indices[1]]
    elif len(X.shape) == 2:
        featurized_X = X[indices[0], indices[1]]
    else:
        raise ValueError('Provide array of valid shape: (number_of_matrices, size, size).')
    if return_df:
        col_names = ['edge_' + str(i) + '_' + str(j) for i,j in zip(indices[0], indices[1])]
        featurized_X = pd.DataFrame(featurized_X, columns=col_names)
    return featurized_X

def degrees(X, return_df = False):
    if len(X.shape) == 3:
        featurized_X = np.sum(X, axis=1)
        shape = (X.shape[0], X.shape[1])
    elif len(X.shape) == 2:
        featurized_X = np.sum(X, axis=1)
        shape = (1, X.shape[1])
    else:
        raise ValueError('Provide array of valid shape: (number_of_matrices, size, size). ')

    if return_df:
        col_names = ['degree_' + str(i) for i in range(X.shape[1])]
        featurized_X = pd.DataFrame(featurized_X.reshape(shape), columns=col_names)
    return featurized_X

def closeness_centrality(X_in):
    X = X_in.copy()
    n_nodes = X.shape[0]
    A_inv = 1./X
    SPL = scipy.sparse.csgraph.dijkstra(A_inv, directed=False,
            unweighted=False)
    sum_distances_vector = np.sum(SPL, 1)
    cl_c = float(n_nodes - 1)/sum_distances_vector
    featurized_X = cl_c
    return featurized_X

def betweenness_centrality(X_in):
    X = X_in.copy()
    n_nodes = X.shape[0]
    n_nodes = X.shape[0]
    A_inv = 1./X
    G_inv = ig.Graph.Weighted_Adjacency(list(A_inv), mode="UNDIRECTED", attr="weight", loops=False)
    btw = np.array(G_inv.betweenness(weights='weight', directed=False))*2./((n_nodes-1)*(n_nodes-2))
    return btw

def eigenvector_centrality(X):
    G = ig.Graph.Weighted_Adjacency(list(X), mode="UNDIRECTED",
                attr="weight", loops=False)
    eigc = G.eigenvector_centrality(weights='weight', directed=False)
    return np.array(eigc)

def pagerank(X):
    G = ig.Graph.Weighted_Adjacency(list(X), mode="DIRECTED", attr="weight", loops=False)
    return np.array(G.pagerank(weights="weight"))

def efficiency(X_in):
    X = X_in.copy()
    n_nodes = X.shape[0]
    A_inv = 1./X
    SPL = scipy.sparse.csgraph.dijkstra(A_inv, directed=False, unweighted=False)
    inv_SPL_with_inf = 1./SPL
    inv_SPL_with_nan = inv_SPL_with_inf.copy()
    inv_SPL_with_nan[np.isinf(inv_SPL_with_inf)]=np.nan
    efs = np.nanmean(inv_SPL_with_nan, 1)
    return efs

def clustering_coefficient(X):
    Gnx = nx.from_numpy_matrix(X)
    clst_geommean = list(nx.clustering(Gnx, weight='weight').values())
    clst_geommean
    return np.array(clst_geommean)

def triangles(X):
    clust = clustering_coefficient(X)

    G = ig.Graph.Weighted_Adjacency(list(X), mode="UNDIRECTED",
            attr="weight", loops=False)
    non_weighted_degrees = np.array(G.degree())
    non_weighted_deg_by_deg_minus_one = np.multiply(non_weighted_degrees,
            (non_weighted_degrees - 1))
    tr = np.multiply(np.array(clust),
            np.array(non_weighted_deg_by_deg_minus_one, dtype = float))/2.
    return tr

def get_cortical(matrix, con):
    ind_dict = {}
    ind_dict['con_aparc150+subcort'] = np.array([item for item in list(range(164))[:-14] if item != 41 and item != 116])
    ind_dict['con_aparc68+subcort'] = np.array([item for item in list(range(84))[:-14] if item != 3 and item != 38])
    ind_dict['con_ROIv_scale33'] = np.array(list(range(34)) + list(range(41,75)))
    ind_dict['con_ROIv_scale60'] = np.array(list(range(57)) + list(range(64,121)))
    ind_dict['con_ROIv_scale125'] = np.array(list(range(108)) + list(range(115,226)))
    ind_dict['con_ROIv_scale250'] = np.array(list(range(223)) + list(range(230,455)))
    ind_dict['con_ROIv_scale500'] = np.array(list(range(501)) + list(range(508,1007)))
    if con in ind_dict.keys():
        inds = ind_dict[con]
        return matrix[inds][:,inds]
    else:
        return matrix

def generate_even_sample(data, n = 1000, seed = 0):
    sample_of_1 = data[data.are_same == 1].sample(n=n, random_state=seed)
    sample_of_0 = data[data.are_same == 0].sample(n=n, random_state=seed)
    return pd.concat([sample_of_1, sample_of_0], axis=0)

################
# Transformers #
################

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin

class DataTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, dataset_path,
                       tractography,
                       model,
                       con,
                       subcortical):

        self.dataset_path   = dataset_path
        self.tractography   = tractography
        self.model          = model
        self.con            = con
        self.subcortical    = subcortical

    def fit(self, gender_df, y=None, **fit_params):
        return self

    def transform(self, gender_df):
        connectomes_dir = '{}trk_processing/track_{}_{}'.format(self.dataset_path, self.tractography, self.model)
        file_ids = np.unique(gender_df['full_id'])
        matrices = {}
        bad_ids = []
        for ID in sorted(file_ids):
            a_connectome_dir = '{}/{}/{}/'.format(connectomes_dir, ID, self.con)
            connectome_files = os.listdir(a_connectome_dir)
            for file in sorted(os.listdir(a_connectome_dir)):
                if 'NORM' not in file and '@' not in file and 'iber' not in file and 'label' not in file:
                    matrix = np.genfromtxt(a_connectome_dir + file)
                    matrix = get_cortical(matrix, self.con)
                    np.fill_diagonal(matrix, 0)
                    matrices[ID] = matrix
        return {
            'gender_df': gender_df,
            'matrices': matrices,
            'y': gender_df.gender,
            'bad_ids': bad_ids
        }

class MatrixNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, norm, copy=True):
        self.norm    = norm
        self.copy    = copy

    def fit(self, data, y=None, **fit_params):
        return self

    def transform(self, data):
        matrices_transformed = {}

        for key in data['matrices'].keys():
            matrices_transformed[key] = self.norm(data['matrices'][key])

        data['matrices'] = matrices_transformed

        return data

class MatrixFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, features, copy=True):
        self.features = features
        self.copy = copy

    def fit(self, data, y=None, **fit_params):
        return self

    def transform(self, data):

        cur_features = {}
        for key in data['matrices'].keys():
            cur_features[key] = self.features[0](data['matrices'][key])
            for feature_func in self.features[1:]:
                cur_features[key] = np.append(cur_features[key], feature_func(data['matrices'][key]))
        data['features'] = cur_features
        matrices = data['gender_df'].full_id.apply(lambda x: data['features'][x])

        matrices = np.array([np.array(values) for values in matrices])

        return np.array(matrices)

    ########
    # Main #
    ########

if __name__ == "__main__":

    #####################
    # FIXES FOR PROBLEM #
    #####################

    to_parse = sys.argv[1]
    dataset, norm, tractography, model, con = to_parse.split('-')

    print('Dataset -- {},\nNormalization -- {},\nTractography -- {},\nmodel -- {},\ncon -- {}.'.format(
        dataset, norm, tractography, model, con))

    overall_path = os.path.abspath('..')
    dataset_path    = '{}/data/{}/'.format(overall_path, dataset)

    gender_df = pd.read_csv(overall_path + '/data/metadata/' + dataset + '_gender_df.csv', index_col=0)
    target = gender_df['gender']
    groups = gender_df['short_id']

    ######################
    # PIPELINER SETTINGS #
    ######################

    grid_cv = GroupKFold(n_splits=5)
    grid_cv = list(grid_cv.split(gender_df, 
                                 target, 
                                 groups))

    eval_cv = GroupShuffleSplit(n_splits=50,
                                test_size=0.2,
                                random_state=0)

    eval_cv = list(eval_cv.split(gender_df, 
                                 target, 
                                 groups))

    data = DataTransformer(dataset_path,
                           tractography,
                           model,
                           con,
                           True).fit_transform(gender_df)

    y = target
    X = data

    norms_dict = {
        'binar'   : [ ('binar',   MatrixNormalizer(binar_norm)) ],
        'max'     : [ ('max',     MatrixNormalizer(max_norm)) ],
        'mean'    : [ ('mean',    MatrixNormalizer(mean_norm)) ],
        'no_norm' : [ ('no_norm', MatrixNormalizer(no_norm)) ]
    }

    if norm not in norms_dict.keys():
        print( '\nUnknown norm. You can use: \
               \n   binar   \
               \n   max     \
               \n   mean    \
               \n   no_norm.')
        raise
    else:
        normalizers = norms_dict[norm]

    featurizers = [
        ('bag_of_edges',          MatrixFeaturizer([bag_of_edges])),
        ('degrees',               MatrixFeaturizer([degrees])),
        ('closeness_centrality',  MatrixFeaturizer([closeness_centrality])),
        ('betweenness_centrality',MatrixFeaturizer([betweenness_centrality])),
        ('eigenvector_centrality',MatrixFeaturizer([eigenvector_centrality])),
        ('pagerank',              MatrixFeaturizer([pagerank])),
        ('efficiency',            MatrixFeaturizer([efficiency])),
        ('clustering_coefficient',MatrixFeaturizer([clustering_coefficient])),
        ('triangles',             MatrixFeaturizer([triangles]))
    ]

    scalers = [
        ('standard', StandardScaler())
    ]

    classifiers = [
        ('LR', LogisticRegression()),
        ('SGD', SGDClassifier())
    ]

    steps = [
        ('Normalizer', normalizers),
        ('Featurizer', featurizers),
        ('Scaler', scalers),
        ('Classifier', classifiers)
    ]

    param_grid = dict(

        LR={'penalty': ['l1', 'l2'],
                'C':[0.01] + [0.05*i for i in range(1,20)],
                'fit_intercept': [True],
                'max_iter': [100, 200],
                'random_state': [0],
                'solver': ['liblinear'],
                'n_jobs': [-1] }
            ,
        SGD= { 'loss':['hinge', 'log', 'modified_huber'],
                'penalty': ['elasticnet'],
                'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
                'l1_ratio': [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                'fit_intercept': [True],
                'n_iter': [100, 200],
                'shuffle': [True],
                'verbose':[0],
                'epsilon': [0.1],
                'n_jobs': [-1],
                'random_state':[0],
                'learning_rate': ['optimal'],
                'eta0': [0.0],
                'power_t': [0.5],
                'class_weight': [None] })

    pipe = Pipeliner(steps, 
                     grid_cv=grid_cv, 
                     eval_cv=eval_cv, 
                     param_grid=param_grid)

    results = pipe.get_results(X=X,
                               y=y,
                               caching_steps=['Normalizer', 
                                              'Featurizer'],
                               scoring=['accuracy', 
                                        'roc_auc'])

    results_name = '_'.join([dataset, tractography, model, con]) + '.csv'
    results_name_file = '_'.join([dataset, norm, tractography, model, con]) + '.csv'
    folder = overall_path + '/results/script_results/gender_results/'
    results.insert(0, 'Dataset',  results_name[:-4])

    if not os.path.exists(folder):
        os.makedirs(folder)

    results.to_csv(folder + results_name_file)
    print('Saved results to {}.'.format(folder + results_name_file))
    print("Done.")

