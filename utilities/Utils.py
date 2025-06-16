import math
import re
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any, TypeVar, Type
from enum import Enum
from scipy.stats import norm, t
import json
from PIL import Image, ImageDraw, ImageFont, ImageOps
from dataclasses import dataclass

from utilities.Pair import Pair
from utilities.Vector2D import Vector2D

T = TypeVar('T')

class Utils:

    @staticmethod
    def string_to_color(c: str) -> Optional[Tuple[int, int, int]]:
        c = c.lower()
        color_map = {
            "blue": (0, 0, 255),
            "black": (0, 0, 0),
            "yellow": (255, 255, 0),
            "red": (255, 0, 0),
            "green": (30, 108, 47),
            "white": (255, 255, 255),
            "brown": (69, 29, 26),
            "pink": (255, 175, 175),
            "orange": (255, 165, 0),
            "light green": (0, 255, 0),
            "purple": (143, 77, 175)
        }
        return color_map.get(c)

    @staticmethod
    def index_of(array: List[str], obj: str) -> int:
        try:
            return array.index(obj)
        except ValueError:
            return -1

    @staticmethod
    def index_of_int(array: List[int], obj: int) -> int:
        try:
            return array.index(obj)
        except ValueError:
            return -1

    @staticmethod
    def games_per_matchup(n_players: int, n_agents: int, total_game_budget: int, self_play: bool) -> int:
        permutations = Utils.player_permutations(n_players, n_agents, self_play)
        return total_game_budget // permutations

    @staticmethod
    def player_permutations(n_players: int, n_agents: int, self_play: bool) -> int:
        if self_play:
            return n_agents ** n_players
        else:
            ret = 1
            for i in range(n_players):
                ret *= (n_agents - i)
            return ret

    @staticmethod
    def generate_permutations(n: int, elements: List[int], all_perms: List[List[int]]) -> None:
        if n == 1:
            all_perms.append(elements.copy())
        else:
            for i in range(n - 1):
                Utils.generate_permutations(n - 1, elements, all_perms)
                if n % 2 == 0:
                    elements[i], elements[n-1] = elements[n-1], elements[i]
                else:
                    elements[0], elements[n-1] = elements[n-1], elements[0]
            Utils.generate_permutations(n - 1, elements, all_perms)

    @staticmethod
    def get_neighbourhood(x: int, y: int, width: int, height: int, way8: bool) -> List[Vector2D]:
        neighbours = []
        # Add orthogonal neighbours
        if x > 0: neighbours.append(Vector2D(x-1, y))
        if x < width - 1: neighbours.append(Vector2D(x+1, y))
        if y > 0: neighbours.append(Vector2D(x, y-1))
        if y < height - 1: neighbours.append(Vector2D(x, y+1))

        # Add diagonal neighbours
        if way8:
            if x > 0 and y > 0: neighbours.append(Vector2D(x-1, y-1))
            if x < width - 1 and y < height - 1: neighbours.append(Vector2D(x+1, y+1))
            if x > 0 and y < height - 1: neighbours.append(Vector2D(x-1, y+1))
            if x < width - 1 and y > 0: neighbours.append(Vector2D(x+1, y-1))
        return neighbours

    @staticmethod
    def normalise(value: float, min_val: float, max_val: float) -> float:
        if min_val < max_val:
            return (value - min_val) / (max_val - min_val)
        elif min_val == max_val:
            return 0.0
        raise ValueError(f"Invalid args in Utils.normalise() - {value} is not in range [{min_val}, {max_val}]")

    @staticmethod
    def noise(input_val: float, epsilon: float, random_val: float) -> float:
        return (input_val + epsilon) * (1.0 + epsilon * (random_val - 0.5))

    @staticmethod
    def sample_from(probabilities: List[float], random_val: float) -> int:
        cdf = 0.0
        for i, prob in enumerate(probabilities):
            cdf += prob
            if cdf >= random_val:
                return i
        raise AssertionError("Should never get here!")

    @staticmethod
    def pdf(potentials: List[float]) -> List[float]:
        sum_pot = sum(potentials)
        if math.isnan(sum_pot) or math.isinf(sum_pot) or sum_pot <= 0.0:
            return [1.0/len(potentials) for _ in potentials]
        if any(p < 0.0 for p in potentials):
            raise ValueError("Negative potential in pdf")
        return [p/sum_pot for p in potentials]

    @staticmethod
    def exponentiate_potentials(potentials: List[float], temperature: float) -> List[float]:
        largest_potential = max(potentials)
        return [math.exp((p - largest_potential) / temperature) for p in potentials]

    @staticmethod
    def clamp(value: float, min_val: float, max_val: float) -> float:
        return max(min_val, min(value, max_val))

    @staticmethod
    def decay(pair: Pair, gamma: float) -> Pair:
        if 0.0 <= gamma < 1.0:
            if pair.a == 0:
                return Pair(0, 0.0)
            new_count = int(pair.a * gamma)
            new_value = pair.b * new_count / pair.a
            return Pair(new_count, new_value)
        return pair

    @staticmethod
    def decay_map(map_data: Dict[Any, Pair], gamma: float) -> Dict[Any, Pair]:
        return {key: Utils.decay(value, gamma) for key, value in map_data.items()}

    @staticmethod
    def get_arg(args: Any, name: str, default_value: T) -> T:
        if isinstance(args, dict):
            return Utils.get_arg_dict(args, name, default_value)
        elif isinstance(args, list):
            return Utils.get_arg_list(args, name, default_value)
        else:
            raise ValueError(f"Unknown args type {type(args)}")

    @staticmethod
    def get_arg_list(args: List[str], name: str, default_value: T) -> T:
        name_lower = name.lower()
        for arg in args:
            if arg.lower().startswith(name_lower + "="):
                parts = arg.split("=")
                if len(parts) > 1:
                    raw_string = parts[1]
                    if isinstance(default_value, Enum):
                        for constant in type(default_value):
                            if constant.name.lower() == raw_string.lower():
                                return constant
                    elif isinstance(default_value, int):
                        return int(raw_string)
                    elif isinstance(default_value, float):
                        return float(raw_string)
                    elif isinstance(default_value, bool):
                        return raw_string.lower() == "true"
                    elif isinstance(default_value, str):
                        return raw_string
                    elif isinstance(default_value, (list, dict)):
                        return json.loads(raw_string)
                    else:
                        raise AssertionError(f"Unexpected type of defaultValue: {type(default_value)}")
                else:
                    print(f"No value specified for {name}, using default value of {default_value}")
        return default_value

    @staticmethod
    def get_arg_dict(args: Dict[str, Any], name: str, default_value: T) -> T:
        if name in args:
            raw_object = args[name]
            if isinstance(default_value, Enum):
                for constant in type(default_value):
                    if constant.name.lower() == str(raw_object).lower():
                        return constant
            elif isinstance(default_value, int):
                return int(raw_object)
            elif isinstance(default_value, (float, bool, str)):
                return raw_object
            else:
                raise AssertionError(f"Unexpected type of defaultValue: {type(default_value)}")
        return default_value

    @staticmethod
    def create_directory(nested_directories: List[str]) -> str:
        folder = ""
        for nested_dir in nested_directories:
            folder = os.path.join(folder, nested_dir)
            os.makedirs(folder, exist_ok=True)
        return folder

    @staticmethod
    def create_directory_from_path(full_directory_path: str) -> str:
        nested_directories = full_directory_path.split(os.sep)
        return Utils.create_directory(nested_directories)

    @staticmethod
    def combination_util(arr: List[int], data: List[int], start: int, end: int, 
                        index: int, r: int, all_data: List[List[int]]) -> None:
        if index == r:
            all_data.append(data.copy())
            return
        if len(all_data) > 1000:
            return

        for i in range(start, end + 1):
            if end - i + 1 >= r - index:
                data[index] = arr[i]
                Utils.combination_util(arr, data, i + 1, end, index + 1, r, all_data)

    @staticmethod
    def combination_util_obj(arr: List[Any], data: List[Any], start: int, end: int, 
                            index: int, r: int, all_data: Set[Tuple[Any, ...]]) -> None:
        if index == r:
            all_data.add(tuple(data.copy()))
            return

        for i in range(start, end + 1):
            if end - i + 1 >= r - index:
                data[index] = arr[i]
                Utils.combination_util_obj(arr, data, i + 1, end, index + 1, r, all_data)

    @staticmethod
    def generate_combinations(arr: List[int], r: int) -> List[List[int]]:
        data = [0] * r
        all_data = []
        Utils.combination_util(arr, data, 0, len(arr) - 1, 0, r, all_data)
        return all_data

    @staticmethod
    def generate_combinations_obj(arr: List[Any], min_size: int, max_size: int) -> Set[Tuple[Any, ...]]:
        all_data = set()
        min_size = max(1, min_size)
        max_size = min(len(arr), max_size)
        for r in range(min_size, max_size + 1):
            data = [None] * r
            Utils.combination_util_obj(arr, data, 0, len(arr) - 1, 0, r, all_data)
        return all_data

    @staticmethod
    def generate_combinations_from_arrays(arr: List[List[Any]]) -> List[List[Any]]:
        combinations = []
        n = len(arr)
        indices = [0] * n

        while True:
            # Add current combination
            combination = [arr[i][indices[i]] for i in range(n)]
            combinations.append(combination)

            # Find rightmost array that has more elements
            next_idx = n - 1
            while next_idx >= 0 and (indices[next_idx] + 1 >= len(arr[next_idx])):
                next_idx -= 1

            if next_idx < 0:
                return combinations

            indices[next_idx] += 1
            for i in range(next_idx + 1, n):
                indices[i] = 0

    @staticmethod
    def mean_diff_standard_error(sum1: float, sum2: float, sum_sq1: float, sum_sq2: float, n1: int, n2: int) -> float:
        mean1 = sum1 / n1
        mean2 = sum2 / n2
        variance1 = sum_sq1 / n1 - mean1 * mean1
        variance2 = sum_sq2 / n2 - mean2 * mean2
        pooled_variance = ((n1 - 1) * variance1 + (n2 - 1) * variance2) / (n1 + n2 - 2)
        return math.sqrt(pooled_variance * (1.0 / n1 + 1.0 / n2))

    @staticmethod
    def standard_z_score(alpha: float, N: int) -> float:
        adjusted_alpha = 1.0 - math.pow(1.0 - alpha, 1.0 / N)
        return norm.ppf(1.0 - adjusted_alpha)

    @staticmethod
    def standard_t_score(alpha: float, N: int, df: int) -> float:
        adjusted_alpha = 1.0 - math.pow(1.0 - alpha, 1.0 / N)
        return t.ppf(1.0 - adjusted_alpha, df)

    @staticmethod
    def search_enum(enum_constants: List[Any], search: str) -> Optional[Any]:
        for obj in enum_constants:
            if str(obj).lower() == search.lower():
                return obj
        return None

    @staticmethod
    def convert_to_type(source_image: Image.Image, target_type: str) -> Image.Image:
        if source_image.mode == target_type:
            return source_image.copy()
        else:
            return source_image.convert(target_type)

    @staticmethod
    def component_to_image(component) -> Image.Image:
        # This would need to be implemented based on your GUI framework
        # Example implementation would vary based on whether you're using tkinter, PyQt, etc.
        raise NotImplementedError("component_to_image not implemented")

    @staticmethod
    def split_camel_case_string(s: str) -> str:
        words = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s)).split()
        return ' '.join(words).strip()

    @staticmethod
    def rotate_image(image: Image.Image, scaled_width_height: Pair[int, int], orientation: int) -> Image.Image:
        degrees = 90 * orientation
        rotated = image.rotate(-degrees, expand=True)
        return rotated.resize((scaled_width_height.a, scaled_width_height.b))

    @staticmethod
    def search_enum_class(enumeration: Type[Enum], search: str) -> Optional[Enum]:
        for enum_value in enumeration:
            if enum_value.name.lower() == search.lower():
                return enum_value
        return None

    @staticmethod
    def get_number_suffix(n: int) -> str:
        if 11 <= n <= 13:
            return "th"
        return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

    @staticmethod
    def enum_to_one_hot(e: Enum, value: float = 1.0) -> List[float]:
        ret_value = [0.0] * len(type(e))
        ret_value[e.value] = value
        return ret_value

    @staticmethod
    def enum_names(e: Type[Enum]) -> List[str]:
        return [enum_value.name for enum_value in e]

    @staticmethod
    def enum_names_from_instance(e: Enum) -> List[str]:
        return Utils.enum_names(type(e))