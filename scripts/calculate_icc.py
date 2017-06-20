#!/usr/bin/python

import numpy as np
import os
import sys
import pandas as pd
from scipy.spatial.distance import *
from scipy.sparse.csgraph import dijkstra, shortest_path, connected_components, laplacian

from sklearn.base import  BaseEstimator, TransformerMixin
from copy import deepcopy

from sklearn.model_selection import StratifiedKFold, cross_val_score, StratifiedShuffleSplit
import networkx as nx
import igraph as ig
import scipy
import time

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.externals.joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import check_scoring
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin, BaseEstimator

from collections import OrderedDict
from itertools import product
from pandas import DataFrame
from pickle import dump, load
from numpy import mean, std, hstack, zeros
from time import time

import os

from rpy2.robjects.numpy2ri import activate
from rpy2.robjects import packages
activate()

psych = packages.importr('psych')


def ICC_rep_anova(data):
    return psych.ICC(data)[0][1][2]

##############################
# Pipeliner rewrited for icc #
##############################


class Pipeliner(object):
    def __init__(self, steps, eval_cv=None, grid_cv=None, param_grid=dict(),
            banned_combos=list()):
        steps = OrderedDict(steps)
        columns = list(steps)
        for column in columns:
            steps[column] = OrderedDict(steps[column])

        def accept_from_banned_combos(row_keys, banned_combo):
            if set(banned_combo) - set(row_keys) == set():
                return False
            else:
                return True

        column_keys = [list(steps[column]) for column in columns]
        plan_rows = list()
        for row_keys in product(*column_keys):
            accept = list()
            for bnnd_cmb in banned_combos:
                accept += [accept_from_banned_combos(row_keys, bnnd_cmb)]

            if all(accept):
                row_of_plan = OrderedDict()
                for column, row_key in zip(columns, row_keys):
                    row_of_plan[column] = row_key
                plan_rows.append(row_of_plan)

        self.plan_table = DataFrame().from_dict(plan_rows)[columns]
        self.named_steps = steps
        self.eval_cv = eval_cv
        self.grid_cv = grid_cv
        self.param_grid = param_grid
        self._cached_data = OrderedDict()
        self.best_params = dict()
        self.scores = dict()

    def transform_with_caching(self, data, row_keys):
        columns = list(self.plan_table.columns[:len(row_keys)])

        def remove_unmatched_caching_data(row_keys):
            cached_keys = list(self._cached_data)
            unmatched_caching_keys = cached_keys.copy()
            for row_key, cached_key in zip(row_keys, cached_keys):
                if not row_key == cached_key:
                    break
                unmatched_caching_keys.remove(row_key)

            for unmatched_caching_key in unmatched_caching_keys:
                del self._cached_data[unmatched_caching_key]

        def transform_data_from_last_cached(row_keys, columns):
            prev_key = list(self._cached_data)[-1]
            for row_key, column in zip(row_keys, columns):
                transformer = self.named_steps[column][row_key]
                data = self._cached_data[prev_key]
                self._cached_data[row_key] = transformer.fit_transform(data)
                prev_key = row_key

        if 'init' not in self._cached_data:
            self._cached_data['init'] = data
            transform_data_from_last_cached(row_keys, columns)
        else:
            row_keys = ['init'] + row_keys
            columns = ['init'] + columns
            remove_unmatched_caching_data(row_keys)
            cached_keys = list(self._cached_data)
            cached_keys_length = len(cached_keys)
            for i in range(cached_keys_length):
                del row_keys[0]
                del columns[0]
            transform_data_from_last_cached(row_keys, columns)

        last_cached_key = list(self._cached_data)[-1]

        return self._cached_data[last_cached_key]

    def get_results(self, data, caching_steps=list(), scoring='accuracy',
            results_file='results.csv', logs_file='results.log', collect_n=None):
        if type(scoring) == str:
            scoring = [scoring]

        columns = list(self.plan_table.columns)
        without_caching = [step for step in columns
                                if step not in caching_steps]


        columns += ['icc_mean','icc_std', 'icc_max', 'icc_median', 'icc_values']

        results = DataFrame(columns=columns)

        try:
            os.remove(results_file)
            print('Removed previous results file -- {}.'.format(results_file))
        except:
            print('No previous results found.')
        if results_file != None:
            results.to_csv(results_file)

        columns = list(self.plan_table.columns)
        results[columns] = self.plan_table


        with open(logs_file, 'w+') as logs:
            N = len(self.plan_table.index)
            for idx in self.plan_table.index:
                print('Line: {}/{}'.format(idx + 1, N))
                logs.write('Line: {}/{}\n'.format(idx + 1, N))
                logs.write('{}\n'.format(str(self.plan_table.loc[idx])))
                row = self.plan_table.loc[idx]
                caching_keys = list(row[caching_steps].values)

                time_point = time()
                icc_values = self.transform_with_caching(data, caching_keys)

                print(np.isnan(icc_values).any())

                results.loc[idx]['icc_mean']   = np.nanmean(icc_values)
                results.loc[idx]['icc_std']    = np.nanstd(icc_values)
                results.loc[idx]['icc_max']    = np.nanmax(icc_values)
                results.loc[idx]['icc_median'] = np.nanmedian(icc_values)
                results.loc[idx]['icc_values'] = str(list(icc_values))
                results.loc[[idx]].to_csv(results_file, header=False, mode='a+')

        return results


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
    bin_matrix[bin_matrix > 0.001] = 1
    return bin_matrix

def mean_norm(matrix):
    normed_matrix = matrix / np.mean(matrix)
    return normed_matrix

def double_norm(function, matrix1, matrix2):
    return function(matrix1), function(matrix2)


###############
# Featurizers #
###############



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
    #epsilon = 0.01
    #X[X == 0] = epsilon
    #X[np.diag_indices(X.shape[0])] = 0
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
    #epsilon = 0.01
    #X[X == 0] = epsilon
    #X[np.diag_indices(X.shape[0])] = 0
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
    #epsilon = 0.01
    #X[X == 0] = epsilon
    #X[np.diag_indices(X.shape[0])] = 0
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

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        connectomes_dir = '{}trk_processing/track_{}_{}'.format(self.dataset_path, self.tractography, self.model)
        file_ids = np.unique(pairs_data[['SUBID1', 'SUBID2']])
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
            'pairs_data': pairs_data,
            'matrices': matrices,
            'y': pairs_data.target,
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

        return data


###################################
# Transformer for ICC calculation #
###################################


class ICC_calculator(BaseEstimator, TransformerMixin):
    def __init__(self, agregate=False):
        self.agregate = agregate

    def fit(self, data, y=None, **fit_params):
        return self

    def transform(self, data):

        def get_id(string):
            return int(string[4:11])

        def get_session(string):
            return int(string[ int(string.find('_ses-') + 5) ])

        def get_number_of_features(features_dict):
            key = list(features_dict.keys())[0]
            return len(features_dict[key])

        def get_number_of_ids(features_dict):
            ids_set = set()
            for key in features_dict:
                ids_set.add( get_id(key) )
            return len(ids_set)

        def get_number_of_sessions(features_dict):
            session_set = set()
            for key in features_dict:
                session_set.add( get_session(key) )
            return len(session_set)

        def get_set_of_ids(features_dict):
            ids_set = set()
            for key in features_dict:
                ids_set.add( get_id(key) )
            return ids_set

        D = get_number_of_features(data['features'])
        I = get_number_of_ids(data['features'])
        S = get_number_of_sessions(data['features'])
        cube = np.zeros((D, I, S))
        print(cube.shape)

        # Creation of dictionary for ids
        ID_dict = {}
        for i, ID in enumerate(sorted(get_set_of_ids(data['features']))):
            ID_dict[ID] = i

        # Filling cube
        for key in data['features']:
            session = int(get_session(key)) - 1
            ID = ID_dict[get_id(key)]
            cube[:,ID,session] = data['features'][key]

        icc = np.zeros(D)
        for i in range(D):
            icc[i] = ICC_rep_anova(cube[i])

        data['cube'] = cube

        return np.round(icc, 4)


########
# Main #
########

if __name__ == "__main__":

    #####################
    # FIXES FOR PROBLEM #
    #####################

    to_parse = sys.argv[1]
    dataset, norm, tractography, model, con = to_parse.split('-')

    print('Dataset -- {},\nNorm -- {}, \nTractography -- {},\nModel -- {},\nCon -- {}.'.format(
        dataset, norm, tractography, model, con))

    overall_path = os.path.abspath('..')
    pairs_data_path = '{}/data/metadata/{}_phenotypic_data_pairwise.csv'.format(
            overall_path,
            dataset)
    dataset_path = '{}/data/{}/'.format(overall_path, dataset)

    pairs_data = pd.read_csv(pairs_data_path, index_col=0)
    pairs_data = pairs_data[pairs_data.TIME_DELTA <= 90]
    n = pairs_data.are_same.sum()
    pairs_data = generate_even_sample(pairs_data, n)
    pairs_data['target'] = pairs_data.are_same

    ######################
    # PIPELINER SETTINGS #
    ######################

    grid_cv = StratifiedKFold(n_splits=5, 
                              shuffle=True,  
                              random_state=0)

    eval_cv = StratifiedShuffleSplit(n_splits=50,
                                     test_size=0.2,
                                     random_state=0)

    grid_cv = StratifiedKFold(n_splits=5, 
                              shuffle=True,  
                              random_state=0)

    eval_cv = StratifiedShuffleSplit(n_splits=50,
                                     test_size=0.2,
                                     random_state=0)

    data = DataTransformer(dataset_path,
                           tractography,
                           model,
                           con,
                           True).fit_transform(pairs_data)

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

    icc = [
        ('icc', ICC_calculator())
    ]


    steps = [
        ('Normalizer', normalizers),
        ('Featurizer', featurizers),
        ('ICC', icc)
    ]


    pipe = Pipeliner(steps, grid_cv=grid_cv, eval_cv=eval_cv)
    results = pipe.get_results(data=data,
                               caching_steps=['Normalizer',
                                              'Featurizer',
                                              'ICC'])

    results_name = '_'.join([dataset, norm, tractography, model, con]) + '.csv'
    results_name_table = '_'.join([dataset, tractography, model, con]) + '.csv'
    folder = overall_path + '/results/script_results/icc_results/'
    results.insert(0, 'Dataset', results_name_table[:-4])

    if not os.path.exists(folder + dataset + '/'):
        os.makedirs(folder + dataset + '/')

    results.to_csv(folder + dataset + '/' + results_name)
    print('Saved results to {}.'.format(folder + dataset + '/' + results_name))
    print("Done.")
