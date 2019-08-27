import pandas as pd
import numpy as np

# Make a dictionary
dict1 = {'Apple': '1, 2, 3, 4',
         'Banana': '1, 2, 3, 4',
         'Pear': '1, 2, 3, 4'}

# Turn the dictionary into a pandas DF
# Add a new columm to the data frame called 'Peach' with nan values
# Modify the first 2 nan values into '1, 2' but leave the rest as nan
