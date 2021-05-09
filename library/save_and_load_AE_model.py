import numpy as np
from library.class_AE import get_compiled_model
import os
# from pathlib import Path
cur_path = os.path.abspath(os.getcwd())


def save_my_model(model, folder_name):
    """ Create the subfolder 'folder_name' in the folder 'results' """
    path_to_it = os.path.join(cur_path, 'results/' + folder_name)
    if not os.path.exists(path_to_it):  # check if the subfolder exists
        os.mkdir(path_to_it)  # create the subfolder
    model.save_weights(path_to_it + '/model', save_format='tf')
    return None


def load_my_model(n_input, folder_name):
    path_to_my_model = os.path.join(cur_path,
                                    'results/' + folder_name + '/model')
    XX = np.random.rand(1, n_input)
    """ initiate a new model before loading the saved model"""
    loaded_model = get_compiled_model(n_input)
    loaded_model.fit(XX, XX, verbose=False)
    """ we will load the saved model to the newly-created model """
    loaded_model.load_weights(path_to_my_model)
    return loaded_model
