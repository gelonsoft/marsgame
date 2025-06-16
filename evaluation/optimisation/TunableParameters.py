import json
import re
from typing import Dict, List, Optional, Any, TypeVar, Type, Union
from core import AbstractParameters
from core.interfaces import ITunableParameters
from utilities import JSONUtils

T = TypeVar('T')

class TunableParameters(AbstractParameters, ITunableParameters[T]):
    debug = False

    def __init__(self):
        super().__init__()
        self.raw_json = None
        self.reset_on = True
        self.static_parameters: List[str] = []
        self.parameter_names: List[str] = []
        self.possible_values: Dict[str, List[Any]] = {}
        self.default_values: Dict[str, Any] = {}
        self.current_values: Dict[str, Any] = {}
        self.parameter_types: Dict[str, type] = {}

    @staticmethod
    def load_from_json_file(params: 'TunableParameters', filename: str) -> None:
        try:
            with open(filename, 'r') as file:
                raw_data = json.load(file)
                TunableParameters.load_from_json(params, raw_data)
        except Exception as e:
            raise AssertionError(f"{type(e).__name__} : {str(e)} : problem loading TunableParameters from file {filename}")

    @staticmethod
    def load_from_json(params: 'TunableParameters', raw_data: Dict[str, Any]) -> None:
        all_params = params.get_parameter_names()

        for p_name in params.static_parameters:
            params.current_values[p_name] = raw_data.get(p_name, params.get_default_parameter_value(p_name))

        for p_name in all_params:
            if TunableParameters.debug:
                print(f"\tLoading {p_name}")
            if TunableParameters.is_param_array(p_name, raw_data):
                p_value = TunableParameters.get_param_list(p_name, raw_data, params.get_default_parameter_value(p_name))
                params.add_tunable_parameter(p_name, params.get_default_parameter_value(p_name), list(p_value))
            else:
                p_value = TunableParameters.get_param(p_name, raw_data, params.get_default_parameter_value(p_name), params)
                if p_value is not None:
                    params.add_tunable_parameter(p_name, p_value)

        params._reset()
        params.raw_json = raw_data
        all_params.extend(["class", "args"])

        for key in raw_data.keys():
            if isinstance(key, str) and key not in all_params and key not in params.static_parameters:
                print(f"Unexpected key in JSON for TunableParameters : {key}")

    @staticmethod
    def get_param(name: str, json_data: Dict[str, Any], default_value: T, params: 'TunableParameters') -> T:
        final_data = json_data.get(name, default_value)
        if final_data is None:
            return None

        data = int(final_data) if isinstance(final_data, int) and not isinstance(final_data, bool) else final_data

        if isinstance(final_data, dict):
            ret_value = JSONUtils.load_class_from_json(final_data)
            if isinstance(ret_value, TunableParameters):
                params.set_parameter_value(name, ret_value)
            return ret_value

        required_class = params.get_parameter_types().get(name)
        if required_class is not None and isinstance(data, required_class):
            return data
        if isinstance(data, int) and required_class == float:
            return float(data)
        if isinstance(data, str) and required_class.__module__ == 'enum':
            for enum_value in required_class:
                if enum_value.name == data:
                    return enum_value
            raise AssertionError(f"No Enum match found for {name} [{data}] in {list(required_class)}")

        print(f"Warning: parsing param {name}; couldn't find correct type, assigning default value: {default_value}")
        return default_value

    @staticmethod
    def is_param_array(name: str, json_data: Dict[str, Any]) -> bool:
        return isinstance(json_data.get(name), list)

    @staticmethod
    def is_param_json(name: str, json_data: Dict[str, Any]) -> bool:
        return isinstance(json_data.get(name), dict)

    @staticmethod
    def get_param_list(name: str, json_data: Dict[str, Any], default_value: T) -> List[T]:
        data = json_data.get(name, default_value)
        if not isinstance(data, list):
            raise AssertionError(f"JSON does not contain an Array as expected for {name}")
        return data

    def copy(self) -> 'TunableParameters':
        ret_value = super().copy()
        tunable = ret_value
        tunable.parameter_names = list(self.parameter_names)
        tunable.possible_values = dict(self.possible_values)
        tunable.default_values = dict(self.default_values)
        tunable.parameter_types = dict(self.parameter_types)
        tunable.reset_on = False

        for name in self.parameter_names:
            value = self.get_parameter_value(name)
            if isinstance(value, TunableParameters):
                sub_params_copy = value.copy()
                tunable.set_parameter_value(name, sub_params_copy)
            else:
                tunable.set_parameter_value(name, value)

        tunable.reset_on = True
        tunable._reset()
        return tunable

    def shallow_copy(self) -> 'TunableParameters':
        ret_value = super().copy()
        tunable = ret_value
        tunable.parameter_names = self.parameter_names
        tunable.possible_values = self.possible_values
        tunable.default_values = self.default_values
        tunable.parameter_types = self.parameter_types
        tunable.current_values = self.current_values
        tunable._reset()
        return tunable

    def add_static_parameter(self, name: str, default_value: T) -> None:
        self.static_parameters.append(name)
        self.default_values[name] = default_value
        self.current_values[name] = default_value

    def add_tunable_parameter(self, name: str, value: T) -> None:
        self.add_tunable_parameter_with_values(name, value, [value])

    def add_tunable_parameter_with_values(self, name: str, default_value: T, all_settings: List[T]) -> None:
        if name not in self.parameter_names:
            self.parameter_names.append(name)
        self.default_values[name] = default_value
        self.parameter_types[name] = type(default_value)
        self.possible_values[name] = list(all_settings)
        self.current_values[name] = default_value

    def add_tunable_parameter_with_class(self, name: str, parameter_class: Type[T], default_value: T, all_settings: List[T]) -> None:
        if name not in self.parameter_names:
            self.parameter_names.append(name)
        self.default_values[name] = default_value
        self.parameter_types[name] = parameter_class
        self.possible_values[name] = list(all_settings)
        self.current_values[name] = default_value

    def add_tunable_parameter_class_only(self, name: str, class_type: Type[T]) -> None:
        if name not in self.parameter_names:
            self.parameter_names.append(name)
        self.default_values[name] = None
        self.parameter_types[name] = class_type
        self.possible_values[name] = []
        self.current_values[name] = None

    def get_parameter_types(self) -> Dict[str, type]:
        return self.parameter_types

    def get_parameter_ids(self) -> List[int]:
        return list(range(len(self.parameter_names)))

    def get_default_parameter_value(self, parameter_name: str) -> Any:
        return self.default_values.get(parameter_name)

    def get_default_override(self, parameter_name: str) -> Any:
        if self.raw_json is not None and parameter_name in self.raw_json and not isinstance(self.raw_json.get(parameter_name), list):
            return self.raw_json.get(parameter_name)
        return self.get_default_parameter_value(parameter_name)

    def set_parameter_value(self, parameter_name: str, value: Any) -> None:
        if "." in parameter_name:
            split = parameter_name.split(".")
            sub_param_name = split[0]
            sub_param = parameter_name[len(sub_param_name) + 1:]
            sub_params = self.get_parameter_value(sub_param_name)
            if isinstance(sub_params, ITunableParameters):
                sub_params.set_parameter_value(sub_param, value)

        param_type = self.parameter_types.get(parameter_name)
        if param_type is not None and param_type.__module__ == 'enum' and isinstance(value, str):
            for enum_value in param_type:
                if enum_value.name == value:
                    self.current_values[parameter_name] = enum_value
                    break
            else:
                raise AssertionError(f"No corresponding Enum found for {value} in {parameter_name}")
        else:
            final_value = int(value) if isinstance(value, int) and not isinstance(value, bool) else value
            self.current_values[parameter_name] = final_value

        if isinstance(value, TunableParameters):
            old_param_names = [n for n in self.parameter_names if n.startswith(f"{parameter_name}.")]
            for name in old_param_names:
                self.parameter_names.remove(name)
                self.possible_values.pop(name, None)
                self.default_values.pop(name, None)
                self.current_values.pop(name, None)

            for name in value.get_parameter_names():
                lifted_name = f"{parameter_name}.{name}"
                self.add_tunable_parameter_with_class(
                    lifted_name,
                    value.get_parameter_types()[name],
                    value.get_default_parameter_value(name),
                    value.get_possible_values(name)
                )
                self.set_parameter_value(lifted_name, value.get_parameter_value(name))

        if self.reset_on:
            self._reset()

    def get_parameter_value(self, parameter_name: str) -> Any:
        return self.current_values.get(parameter_name)

    def get_parameter_name(self, parameter_id: int) -> str:
        return self.parameter_names[parameter_id]

    def get_possible_values(self, name: str) -> List[Any]:
        return list(self.possible_values.get(name, []))

    def get_default_parameter_values(self) -> Dict[str, Any]:
        return dict(self.default_values)

    def instance_from_json(self, json_object: Dict[str, Any]) -> 'ITunableParameters':
        return JSONUtils.load_class_from_json(json_object)

    def set_raw_json(self, json_data: Dict[str, Any]) -> None:
        self.raw_json = json_data

    def instance_to_json(self, exclude_defaults: bool = False, settings: Dict[str, int] = None) -> Dict[str, Any]:
        if settings is None:
            settings = {}
        ret_value = {"class": self.__class__.__name__}
        
        for name in self.parameter_names:
            if "." in name:
                continue
            value = self.get_parameter_value(name)
            if value is not None:
                if isinstance(value, ITunableParameters):
                    sub_settings = {
                        k[len(name) + 1:]: v 
                        for k, v in settings.items() 
                        if "." in k and k.startswith(f"{name}.")
                    }
                    value = value.instance_to_json(exclude_defaults, sub_settings)
                else:
                    if hasattr(value, "name"):  # Enum check
                        value = value.name
                    elif not isinstance(value, (int, float, str, bool)):
                        if self.raw_json is None:
                            if JSONUtils.are_values_equal(value, self.get_default_parameter_value(name)):
                                continue
                            raise AssertionError(f"No raw_json available to extract value for {name}")
                        raw_value = self.raw_json.get(name)
                        if isinstance(raw_value, list) and name in settings:
                            value = raw_value[settings[name]]
                    if exclude_defaults and JSONUtils.are_values_equal(value, self.get_default_parameter_value(name)):
                        continue
                ret_value[name] = value

        if self.raw_json is not None:
            for key in self.raw_json:
                if isinstance(key, str) and key not in settings and key not in ret_value:
                    if exclude_defaults and JSONUtils.are_values_equal(self.raw_json[key], self.get_default_parameter_value(key)):
                        continue
                    ret_value[key] = self.raw_json[key]
        return ret_value

    def get_parameter_values(self) -> Dict[str, Any]:
        return dict(self.current_values)

    def set_parameter_values(self, values: Dict[Any, Any]) -> None:
        for key in values:
            if isinstance(key, str):
                self.set_parameter_value(key, values[key])
            else:
                name = self.get_parameter_name(key)
                self.set_parameter_value(name, values[key])

    def get_parameter_names(self) -> List[str]:
        return list(self.parameter_names)

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, TunableParameters):
            return False
        return (self._equals(o) and
                o.parameter_names == self.parameter_names and
                o.possible_values == self.possible_values and
                o.current_values == self.current_values and
                o.default_values == self.default_values)

    def all_parameters_and_values_equal(self, other: 'TunableParameters') -> bool:
        for name in self.parameter_names:
            if name == "randomSeed":
                continue
            if self.current_values.get(name) is None and other.current_values.get(name) is None:
                continue
            if isinstance(self.current_values.get(name), TunableParameters):
                if not self.current_values[name].all_parameters_and_values_equal(other.current_values[name]):
                    return False
            elif self.current_values.get(name) != other.current_values.get(name):
                return False
        return True

    def __hash__(self) -> int:
        return hash((super().__hash__(), tuple(self.parameter_names), 
                    frozenset(self.possible_values.items()), 
                    frozenset(self.default_values.items()), 
                    frozenset(self.current_values.items())))