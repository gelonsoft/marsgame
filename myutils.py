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