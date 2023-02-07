############# CONTENTS OF THE FILE #############
"""
This file is intended to be a library that can be imported at any time by the 
main scripts. It contains different functions which are used repeatedly over
the ANN training and post-processing. Each function has a documentation 
text. 
"""

############# IMPORTS #############
import re, os
from math import log10, floor

############# FUNCTIONS #############
def chunks(l, n):
    """
    Splits an input list l into chunks of size n.

    Parameters
    ----------
    l : list
        List to be splitted.
    n : int
        Size of each chunk.

    Returns
    -------
    splitted_list: list
        Splitted list.

    """
    n = max(1, n)   
    splitted_list = [l[i:i+n] for i in range(0, len(l), n)]
    return splitted_list

def split(l, n):
    """
    Splits an input list l into n chunks.
    
    Parameters
    ----------
    l : list
        List to be splitted.
    n : TYPE
        Number of chunks.

    Returns
    -------
    splitted_list: list
        Splitted list.

    """
    n = min(n, len(l)) # don't create empty buckets
    k, m = divmod(len(l), n)
    splitted = (l[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    splitted_list = list(splitted)
    return splitted_list

def num_sort(test_string):
    """
    Sorts a given list of strings numerically.

    Parameters
    ----------
    test_string : list
        List of strings.

    Returns
    -------
    sorted_list : list
        Numerically-sorted list.

    """
    sorted_list = list(map(int, re.findall(r'\d+', test_string)))[0]
    
    return sorted_list

def get_keys_from_value(d, val):
    """
    Given a dictionary and a value, it returns the corresponding key.

    Parameters
    ----------
    d : dict
        Input dictionary.
    val : any
        Dictionary value that we want to get the key of.

    Returns
    -------
    keys_list: list
        List of the keys that correspond to 'val'.

    """
    keys_list = [k for k, v in d.items() if v == val]
    return keys_list

def show_layers(model):
    """
    Shows the layers of the input Neural Network model.

    Parameters
    ----------
    model : torch object NN
        NN model.

    Returns
    -------
    None.

    """
    print("\nLayers and parameters:\n")
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : " \
              f"{param[:100]} \n")
        
def dir_support(list_of_nested_dirs):
    """
    Directories support: ensures that the (nested) directories given via the 
    input list do exist, creating them if necessary. 

    Parameters
    ----------
    list_of_nested_dirs : list
        Nested directories in order.

    Returns
    -------
    None.

    """
    for i in range(len(list_of_nested_dirs)):
        potential_dir = '/'.join(list_of_nested_dirs[:i+1]) 
        if not os.path.exists(potential_dir):
            os.makedirs(potential_dir)
            print(f'Creating directory {potential_dir}...')
            
def round_to_1(x):
    """
    Rounds a number to the first decimal place. Useful for computing errors.

    Parameters
    ----------
    x : float
        Number to round.

    Returns
    -------
    rounded_number : float
        Rounded number.

    """
    rounded_number = round(x, -int(floor(log10(abs(x)))))
    return rounded_number

def train_loop(model, train_data, x2_y2_z2, loss_fn, optimizer):
    E, psi = loss_fn(model=model,
                     train_data=train_data, 
                     x2_y2_z2=x2_y2_z2)
    optimizer.zero_grad()
    E.backward()
    optimizer.step()
    
    return E, psi