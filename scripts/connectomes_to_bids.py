#!/usr/bin/python

import sys
import os
import re
from collections import OrderedDict
from shutil import copyfile


def make_inside_path(root_folder=None, 
                     data_type=None, 
                     subject=None, 
                     session=None,
                     model=None,
                     tractography=None,
                     parcellation=None,
                     variant=None,
                     modality=None,
                     extension=None):

    root_folder = '' if root_folder is None else root_folder
    data_type = '' if data_type is None else data_type
    overall_template = ''
    
    keys = ('subject', 'session', 'model',
            'tractography', 'parcellation', 'variant',
            'modality_suffix', 'extension')

    values = (subject, session, model,
              tractography, parcellation, variant, 
              modality, extension)
    
    val_dict = OrderedDict(zip(keys, values))
    
    patterns = ('sub-{}', '_ses-{}', '_model-{}', '_tractography-{}', '_parcellation-{}','_variant-{}', '_{}', '.{}')
    
    flags=[True if val_dict[item] is not None else False for item in keys]
    
    replacements = [patterns[i].format(values[i]) if flags[i] else '' for i in range(len(keys))]
    template_dict = OrderedDict(zip(keys, replacements))
    
    filename = ''.join(replacements)
    
    folder = '/{}/derivatives/MICCAIpipeline/{}/{}/{}/'.format(root_folder, template_dict['subject'],
                                                               template_dict['session'][1:], data_type )
    folder = folder.split('/')
    #print(folder)
    folder = [item for item in folder if item != '']
    folder = '/' + '/'.join(folder) + '/'
    
    return folder + filename


def absoluteFilePaths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))


def resultsToBIDS(expr, data_root, destination_root, parcel_replace_dict):

    N = len(list(absoluteFilePaths(data_root)))

    for i, abspath in enumerate(absoluteFilePaths(data_root)):

        print('{}/{}'.format(i, N), end='\r')

        tractography, model, subject, session, \
                parcellation, subject_check, session_check, normalization = re.findall(expr, abspath)[0]
        
        assert subject == subject_check
        assert session == session_check

        if normalization != 'NORM':
            normalization = None
        
        destination = make_inside_path(root_folder=destination_root,
                                       data_type='dwi',
                                       subject=subject,
                                       session=session,
                                       model=model,
                                       tractography=tractography,
                                       parcellation=parcel_replace_dict[parcellation],
                                       variant=normalization,
                                       modality='connectome',
                                       extension='txt')
        
        dirname = os.path.dirname(destination)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        if not os.path.isfile(destination):
            copyfile(abspath, destination)


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print('Transformes connectomes to BIDS.\n'
              'Permitted datasets as input:\n'
              '\tBNU_1\n'
              '\tHNU_1\n'
              '\tIPCAS_1\n')

    data_roots = {'BNU_1': os.path.abspath('../data/BNU_1'), 
                  'HNU_1': os.path.abspath('../data/HNU_1'), 
                  'IPCAS_1': os.path.abspath('../data/IPCAS_1')}

    destination_roots = {'BNU_1': os.path.abspath('../data/BNU_1_BIDS'),
                         'HNU_1': os.path.abspath('../data/HNU_1_BIDS'),
                         'IPCAS_1': os.path.abspath('../data/IPCAS_1_BIDS')}

    for dataset in sys.argv[1:]:
        
        assert dataset in data_roots.keys(), \
            "Permitted datasets: BNU_1, HNU_1, IPCAS_1."

        expr = 'trk_processing/track_(\w+)_(\w+)/sub-(\w+)_ses-(\w+)/con_(\S+)/sub-(\w+)_ses-(\w+)_NxNmatrix_FULL_(\w+).txt'
        data_root = data_roots[dataset]
        destination_root = destination_roots[dataset]

        parcel_replace_dict ={'ROIv_scale250': 'lausanne250', 
                              'ROIv_scale125': 'lausanne125',
                              'ROIv_scale500': 'lausanne500',
                              'ROIv_scale60':  'lausanne60',
                              'ROIv_scale33':  'lausanne33',
                              'aparc150+subcort': 'destriex',  
                              'aparc68+subcort':  'desikan'}

        resultsToBIDS(expr, data_root, destination_root, parcel_replace_dict)
        print('{}: transformed to BIDS. Look at {}'.format(dataset, destination_root)) 
