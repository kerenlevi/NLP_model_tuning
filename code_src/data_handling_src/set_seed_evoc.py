import argparse
import os

def add_seed_function(package_path):
    init_file_path = os.path.join(package_path, '__init__.py')
    
    # Seed function code to be inserted
    seed_function_code = """
import numpy as np
import numba
import random

def set_seed(seed):
    \"\"\"Set the random seed for reproducibility.\"\"\"
    np.random.seed(seed)
    random.seed(seed)
    
    @numba.njit
    def set_numba_seed(seed):
        np.random.seed(seed)
        random.seed(seed)

    set_numba_seed(seed)
"""
    # Read the content of the __init__.py file
    with open(init_file_path, 'r') as file:
        init_content = file.read()

    if 'def set_seed(' not in init_content:
        init_content = init_content + seed_function_code #+ init_content

    # Ensure set_seed is in __all__
    if "__all__ = [" not in init_content:
        # Add __all__ definition if not present
        init_content += "\n__all__ = ['EVoC', 'evoc_clusters', 'set_seed']\n"
    elif "'set_seed'" not in init_content:
        # Append set_seed to existing __all__ list
        init_content = init_content.replace("__all__ = [", "__all__ = ['set_seed', ")

    with open(init_file_path, 'w') as file:
        file.write(init_content)


def main():
    parser = argparse.ArgumentParser(description='Add seed function to evoc package.')
    parser.add_argument('package_path', type=str, help='Path to the evoc package')
    
    args = parser.parse_args()
    add_seed_function(args.package_path)

if __name__ == '__main__':
    main()
    
    # python script_name.py /home/sagemaker-user/src_code/evoc
 