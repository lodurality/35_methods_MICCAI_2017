import os
import re
from collections import OrderedDict


def make_inside_path(subject=None, session=None, run=None, 
              data_type=None, acquisition=None, 
              task=None, root_folder=None, 
              variant=None, modality=None, 
              extension=None, reconstruction=None):
    
    root_folder = '' if root_folder is None else root_folder
    data_type = '' if data_type is None else data_type
    scope = locals()
    overall_template = ''
    
    keys = ('subject', 'session',  'task',
            'acquisition', 'reconstruction' ,'run', 
            'variant', 'modality_suffix', 'extension')

    values = (subject, session, task,  
              acquisition, reconstruction, run, 
              variant, modality, extension)
    
    val_dict = OrderedDict(zip(keys, values))
    
    patterns = ('sub-{}', '_ses-{}', '_task-{}', '_acq-{}', '_rec-{}','_run-{}', '_variant-{}', '_{}', '.{}')
    
    flags=[True if val_dict[item] is not None else False for item in keys]
    
    replacements = [patterns[i].format(values[i]) if flags[i] else '' for i in range(len(keys))]
    template_dict = OrderedDict(zip(keys, replacements))
    
    filename = ''.join(replacements)
    
    folder = '/{}/{}/{}/{}/'.format(root_folder, template_dict['subject'],
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


expr = 'trk_processing/track_(\w+)_(\w+)/sub-(\w+)_ses-(\w+)/con_(\S+)/sub-(\w+)_ses-(\w+)_NxNmatrix_FULL_(\w+).txt'
data_path = os.path.abspath('../data/BNU_1')

for abspath in absoluteFilePaths(data_path):
    tractography, model, subject, session, \
            atlas, subject_check, session_check, normalization = re.findall(expr, abspath)[0]
    
    assert subject == subject_check
    assert session == session_check

    if normalization != 'NORM':
        normalization = None

    #print(tractography, model, subject, session, atlas, normalization)
    
    destination = make_inside_path(subject=subject,
                                   session=session,
                                   data_type='dwi',
                                   root_folder=data_path,
                                   reconstruction='{}-{}-{}'.format(tractography,
                                                                    model,
                                                                    atlas),
                                   variant=normalization,
                                   modality='connectome',
                                   extension='txt')
    print(destination)
