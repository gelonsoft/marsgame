import numpy as np


def get_stat(source, key_path):
        """Safely get nested dict value using dot-notation keys."""
        keys = key_path.split(".")
        value = source
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, 0)
            else:
                return 0
        return value



def find_first_with_nested_attr(data, target_key, parent=None):
    # Check if target_key is in the current dictionary
    if isinstance(data, dict):
        if target_key in data:
            return (parent,data[target_key])  # Found the dict; return its parent
        
        # Recurse into nested values
        for value in data.values():
            result = find_first_with_nested_attr(value, target_key, parent=data)
            if result is not None:
                return result
                
    # Recurse into list elements (the parent remains the last dict encountered)
    elif isinstance(data, list):
        for item in data:
            result = find_first_with_nested_attr(item, target_key, parent=parent)
            if result is not None:
                return result
                
    return None


def create_fixed_size_array(data_list, fixed_size, dtype=np.float32, pad_value=0.0):
    """
    Creates a numpy array of a fixed size from a list of any size.

    If the list is longer than fixed_size, it is truncated.
    If the list is shorter than fixed_size, it is padded with pad_value.

    Args:
        data_list (list): The input list.
        fixed_size (int): The desired size of the output array.
        dtype (numpy.dtype, optional): The desired data type of the array.
        pad_value (int/float, optional): The value used for padding.

    Returns:
        numpy.ndarray: A numpy array of the specified fixed_size.
    """
    # 1. Truncate the list if it is too long
    truncated_list = data_list[:fixed_size]

    # 2. Pad the list if it is too short
    if len(truncated_list) < fixed_size:
        padding_needed = fixed_size - len(truncated_list)
        # Extend the list with the specified pad_value
        truncated_list.extend([pad_value] * padding_needed)

    # 3. Convert the final fixed-size list to a numpy array
    # Ensure all elements have the same type for the array (especially padding values)
    if dtype is None:
        # Determine dtype automatically from the list, or set a default if empty
        dtype = np.array(truncated_list).dtype
    
    return np.array(truncated_list, dtype=dtype)