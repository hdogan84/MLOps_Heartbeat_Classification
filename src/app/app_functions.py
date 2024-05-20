import numpy as np
import pandas as pd

def select_random_row(X_test, y_test):
    """
    Select a random row from the test dataset and its corresponding target.

    Parameters:
    X_test (pd.DataFrame or np.array): Test features dataset.
    y_test (pd.Series or np.array): Test target dataset.

    Returns:
    tuple: A tuple containing the random row from X_test and its corresponding target from y_test.
    """
    # Ensure the random selection is reproducible
    #np.random.seed(42) #this leads to selection of the same "random" row each time...just for debugging
    
    # Select a random index
    random_index = np.random.randint(0, len(X_test))

    
    # Get the random row and its corresponding target
    rand_row = pd.DataFrame(X_test.iloc[random_index] if hasattr(X_test, 'iloc') else X_test[random_index]).T #transformation is necessary? Is this bullshit and an upstream problem?
    #print("rand_row from app functions:", rand_row)
    rand_target = y_test.iloc[random_index] if hasattr(y_test, 'iloc') else y_test[random_index]
    print("random index for random row:", random_index)
    return rand_row, rand_target