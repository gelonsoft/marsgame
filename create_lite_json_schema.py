import os
import json

def remove_attribute_recursive(obj, attribute_to_remove):
    """
    Removes a specified attribute from a JSON object recursively.

    Args:
        obj: The JSON object (dict or list) to process.
        attribute_to_remove: The name of the attribute to remove.

    Returns:
        The modified JSON object with the attribute removed.
    """
    if isinstance(obj, dict):
        if attribute_to_remove in obj:
            del obj[attribute_to_remove]
        for key, value in obj.items():
            remove_attribute_recursive(value, attribute_to_remove)
    elif isinstance(obj, list):
        for item in obj:
            remove_attribute_recursive(item, attribute_to_remove)
    return obj

def clear_attribute_recursive(obj, attribute_to_clear):
    """
    Removes a specified attribute from a JSON object recursively.

    Args:
        obj: The JSON object (dict or list) to process.
        attribute_to_remove: The name of the attribute to remove.

    Returns:
        The modified JSON object with the attribute removed.
    """
    if isinstance(obj, dict):
        if attribute_to_clear in obj and obj[attribute_to_clear] is str:
            obj[attribute_to_clear]=""
        for key, value in obj.items():
            clear_attribute_recursive(value, attribute_to_clear)
    elif isinstance(obj, list):
        for item in obj:
            clear_attribute_recursive(item, attribute_to_clear)
    return obj

def process_file(source_filename,result_filename):
    with open(source_filename,"rb") as f:
        data=f.read()
        data = json.loads(data)
        data = remove_attribute_recursive(data,"enum")
        data = clear_attribute_recursive(data,"description")
        with open(result_filename,"w") as r:
            r.write(json.dumps(data))
            

process_file("gs.response.schema.formatted.json","gs.response.schema.lite.json")
process_file("gs.schema.formatted.json","gs.schema.lite.json")

