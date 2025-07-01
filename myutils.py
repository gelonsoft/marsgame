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