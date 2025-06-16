import json
import random
from typing import Callable, Collection, Dict, Iterator, Set, Optional, List, TypeVar, Generic, TYPE_CHECKING
from abc import ABC, abstractmethod
from enum import Enum
from core.CoreConstants import CoreConstants, VisibilityMode

from core.properties import PropertyBoolean, PropertyColor, PropertyInt, PropertyIntArray, PropertyIntArrayList, PropertyLong, PropertyLongArray, PropertyLongArrayList, PropertyString, PropertyStringArray, PropertyVector2D
from core.properties.Property import Property
from utilities import Hash
from utilities.Pair import Pair

if TYPE_CHECKING:
    from core.AbstractGameState import AbstractGameState

T = TypeVar('T', bound='Component')

class IComponentContainer(Generic[T]):
    """
    An interface to be used on any Component that contains other Components.
    The interface is 'read-only', and deliberately avoids specifying add/remove type methods, and has two purposes:

    i) To be used to gather information about game states for game metrics and comparisons (see GameReport as example)
    ii) To indicate who can see the contents of the Container (Everyone, No-one, just the Owner)?
    iii) As a holder of a few useful stream-related default methods - these are all read-only methods.

    :param T: The Type of Component that the Container holds
    """

    def get_components(self) -> List[T]:
        """
        :return: A list of all the Components in the Container
        """
        raise NotImplementedError

    def get_visibility_mode(self) -> VisibilityMode:
        raise NotImplementedError

    def stream(self):
        """
        :return: A stream of all the Components in the Container
        """
        return (c for c in self.get_components())

    def get_size(self) -> int:
        """
        :return: the size of this deck (number of components in it).
        """
        return len(self.get_components())

    def sum_double(self, lambda_func: Callable[[T], float]) -> float:
        ret_value = 0.0
        for c in self.get_components():
            ret_value += lambda_func(c)
        return ret_value

    def sum_int(self, lambda_func: Callable[[T], int]) -> int:
        ret_value = 0
        for c in self.get_components():
            ret_value += lambda_func(c)
        return ret_value
    
class ComponentType(Enum):
    pass  # Assuming this is defined elsewhere in the project

class Component(ABC):
    _id = 0  # Class variable to track component IDs

    def __init__(self, component_type: ComponentType, name: Optional[str] = None, component_id: Optional[int] = None):
        """
        Initialize a Component with a type and optional name and ID.
        If no ID is provided, a new one is generated.
        """
        if component_id is None:
            self.component_id = Component._id
            Component._id += 1
        else:
            self.component_id = component_id
        
        self.type = component_type
        self.component_name = name if name is not None else str(component_type)
        self.properties: Dict[int, Property] = {}
        self.owner_id = -1  # Default owner is the game

    @abstractmethod
    def copy(self) -> 'Component':
        """
        Create a copy of this component. To be implemented by subclasses.
        """
        pass

    def copy_with_player(self, player_id: int) -> 'Component':
        """
        Create a copy of this component with a specific owner ID.
        """
        return self.copy()

    def get_type(self) -> ComponentType:
        """
        Get the type of this component.
        """
        return self.type

    def get_num_properties(self) -> int:
        """
        Get the number of properties this component has.
        """
        return len(self.properties)

    def get_owner_id(self) -> int:
        """
        Get the ID of the owner of this component.
        """
        return self.owner_id

    def set_owner_id(self, owner_id: int) -> None:
        """
        Set the ID of the owner of this component.
        """
        self.owner_id = owner_id

    def get_component_id(self) -> int:
        """
        Get the unique ID of this component.
        """
        return self.component_id

    def get_component_name(self) -> str:
        """
        Get the name of this component.
        """
        return self.component_name

    def set_component_name(self, name: str) -> None:
        """
        Set the name of this component.
        """
        self.component_name = name

    def get_properties(self) -> Dict[int, Property]:
        """
        Get all properties of this component.
        """
        return self.properties

    def get_property(self, prop_id: int) -> Optional[Property]:
        """
        Get a property by its ID.
        """
        return self.properties.get(prop_id)

    def get_property_by_hash(self, hash_string: str) -> Optional[Property]:
        """
        Get a property by its hash string.
        """
        return self.properties.get(Hash.get_instance().hash(hash_string))

    def set_property(self, prop: Property) -> None:
        """
        Add or update a property.
        """
        self.properties[prop.get_hash_key()] = prop

    def set_properties(self, props: Dict[int, Property]) -> None:
        """
        Add or update multiple properties.
        """
        for prop in props.values():
            self.set_property(prop)

    @staticmethod
    def parse_component(c: 'Component', obj: Dict, ignore_keys: Optional[Set[str]] = None) -> 'Component':
        """
        Parse a Component from a JSON-like dictionary.
        """
        if ignore_keys is None:
            ignore_keys = set()

        for key, value in obj.items():
            if key in ignore_keys:
                continue

            if isinstance(value, list):  # Assuming JSONArray is represented as a list
                type_str = value[0]
                prop = None

                if "[]" in type_str:  # Array type
                    values = value[1]
                    if "String" in type_str:
                        prop = PropertyStringArray(key, values)
                    elif "Integer" in type_str:
                        prop = PropertyIntArray(key, values)
                    elif "Long" in type_str:
                        prop = PropertyLongArray(key, values)
                elif "<>" in type_str:  # List type
                    values = value[1]
                    if "Integer" in type_str:
                        prop = PropertyIntArrayList(key, values)
                    elif "Long" in type_str:
                        prop = PropertyLongArrayList(key, values)
                else:
                    if "String" in type_str:
                        prop = PropertyString(key, value[1])
                    elif "Color" in type_str:
                        prop = PropertyColor(key, value[1])
                    elif "Vector2D" in type_str:
                        prop = PropertyVector2D(key, value[1])
                    elif "Boolean" in type_str:
                        prop = PropertyBoolean(key, value[1])
                    elif "Integer" in type_str:
                        prop = PropertyInt(key, int(value[1]))
                    elif "Long" in type_str:
                        prop = PropertyLong(key, value[1])

                if prop is not None:
                    c.set_property(prop)

        return c

    def copy_component_to(self, target: 'Component') -> None:
        """
        Copy properties and other attributes to another component.
        """
        target.properties.clear()
        for prop_key, prop in self.properties.items():
            target.set_property(prop.copy())
        target.owner_id = self.owner_id
        target.component_name = self.component_name

    def __str__(self) -> str:
        return f"Component{{component_id={self.component_id}, type={self.type}, owner_id={self.owner_id}, component_name='{self.component_name}', properties={self.properties}}}"

    def to_string(self, player_id: int) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Component):
            return False
        return self.component_id == other.component_id

    def __hash__(self) -> int:
        return self.component_id
    
    
class Deck(Component, Generic[T], IComponentContainer[T]):
    def __init__(self, name: str, visibility: VisibilityMode, owner_id: int = -1, capacity: int = -1, component_id: int = -1):
        super().__init__(component_type=ComponentType.DECK, name=name, component_id=component_id)
        self.components: List[T] = []  # Using list instead of LinkedList
        self.owner_id = owner_id
        self.capacity = capacity
        self.visibility = visibility

    def __iter__(self) -> Iterator[T]:
        return iter(self.components)

    @classmethod
    def load_decks_of_cards(cls, filename: str) -> List['Deck[Card]']:
        decks = []
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                for deck_data in data:
                    new_deck = cls.load_deck_of_cards(deck_data)
                    decks.append(new_deck)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading decks: {e}")
        return decks

    @classmethod
    def load_deck_of_cards(cls, deck_data: dict) -> 'Deck[Card]':
        name = deck_data['name'][1] if isinstance(deck_data['name'], list) else deck_data['name']
        new_deck = cls(name, VisibilityMode.HIDDEN_TO_ALL)
        new_deck.set_visibility(VisibilityMode(deck_data['visibility']))
        for card_data in deck_data['cards']:
            new_card = Card.parse_from_json(card_data)  # Assuming Card has a parse_from_json method
            new_deck.add(new_card)
        return new_deck

    def draw(self) -> Optional[T]:
        return self.pick(0)

    def pick(self, idx: int) -> Optional[T]:
        if 0 <= idx < len(self.components):
            return self.components.pop(idx)
        return None

    def pick_random(self, rng: random.Random = None) -> Optional[T]:
        rng = rng or random
        if self.components:
            return self.pick(rng.randint(0, len(self.components) - 1))
        return None

    def pick_last(self) -> Optional[T]:
        return self.pick(len(self.components) - 1) if self.components else None

    def peek(self, idx: int = 0) -> Optional[T]:
        if 0 <= idx < len(self.components):
            return self.components[idx]
        return None

    def peek_multiple(self, idx: int, amount: int) -> List[T]:
        end_idx = min(idx + amount, len(self.components))
        return self.components[idx:end_idx] if idx < len(self.components) else []

    def add(self, component: T, index: int = 0) -> bool:
        if component is None:
            raise ValueError("Cannot add None to a Deck")
        component.owner_id = self.owner_id
        self.components.insert(index, component)
        return self.capacity == -1 or len(self.components) <= self.capacity

    def add_to_bottom(self, component: T) -> bool:
        return self.add(component, len(self.components))

    def add_deck(self, other_deck: 'Deck[T]', index: int = 0) -> bool:
        for comp in other_deck.components:
            comp.owner_id = self.owner_id
        self.components[index:index] = other_deck.components
        return self.capacity == -1 or len(self.components) <= self.capacity

    def add_collection(self, collection: Collection[T], index: int = 0) -> bool:
        for comp in collection:
            comp.owner_id = self.owner_id
        self.components[index:index] = collection
        return self.capacity == -1 or len(self.components) <= self.capacity

    def remove(self, component: T = None, idx: int = None) -> bool:
        if component is not None:
            component.owner_id = -1
            try:
                self.components.remove(component)
                return True
            except ValueError:
                return False
        elif idx is not None and 0 <= idx < len(self.components):
            self.components[idx].owner_id = -1
            self.components.pop(idx)
            return True
        return False

    def remove_all(self, items: List[T]) -> None:
        for component in items:
            if not self.remove(component):
                raise ValueError(f"{component} not found in {self}")

    def contains(self, component: T) -> bool:
        return component in self.components

    def clear(self) -> None:
        for comp in self.components:
            comp.owner_id = -1
        self.components.clear()

    def shuffle(self, rng: random.Random = None, from_index: int = 0, to_index: int = None) -> None:
        rng = rng or random
        to_index = to_index if to_index is not None else len(self.components)
        sublist = self.components[from_index:to_index]
        rng.shuffle(sublist)
        self.components[from_index:to_index] = sublist

    def get_components(self) -> List[T]:
        return self.components

    def set_components(self, components: List[T]) -> None:
        self.components = components
        for comp in components:
            comp.owner_id = self.owner_id

    def get_capacity(self) -> int:
        return self.capacity

    def set_capacity(self, capacity: int) -> None:
        self.capacity = capacity

    def is_over_capacity(self) -> bool:
        return self.capacity != -1 and len(self.components) > self.capacity

    def set_component(self, idx: int, component: T) -> None:
        component.owner_id = self.owner_id
        self.components[idx] = component

    def __getitem__(self, idx: int) -> T:
        return self.components[idx]

    def get_visibility_mode(self) -> VisibilityMode:
        return self.visibility

    def set_visibility(self, mode: VisibilityMode) -> None:
        self.visibility = mode

    def copy(self) -> 'Deck[T]':
        new_deck = Deck(self.component_name, self.visibility, self.owner_id, self.capacity, self.component_id)
        self._copy_to(new_deck)
        return new_deck

    def _copy_to(self, deck: 'Deck[T]') -> None:
        deck.components = [comp.copy() for comp in self.components]
        deck.capacity = self.capacity
        super()._copy_component_to(deck)

    def _copy_to_for_player(self, deck: 'Deck[T]', player_id: int) -> None:
        deck.components = [comp.copy(player_id) for comp in self.components]
        deck.capacity = self.capacity
        super()._copy_component_to(deck)

    def __str__(self) -> str:
        if not self.components:
            return "EmptyDeck"
        return ",".join(str(el) for el in self.components)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Deck):
            return False
        return (super().__eq__(other) and 
                self.capacity == other.capacity and 
                self.components == other.components)

    def __hash__(self) -> int:
        return hash((self.capacity, self.owner_id, self.component_id, tuple(self.components)))
    
class DeterminisationUtilities:
    """
    Utility class for reshuffling cards in decks based on visibility and predicates.
    """
    
    @staticmethod
    def reshuffle(player: int, decks: List['Deck[T]'], lambda_func: Callable[[T], bool], rnd: random.Random) -> None:
        """
        Reshuffles all cards across the list of decks that meet the predicate and are not visible to player.
        """
        if player < 0:
            return

        # Gather up all unknown cards for reshuffling
        all_cards = Deck("temp", -1, CoreConstants.VisibilityMode.HIDDEN_TO_ALL)

        for d in decks:
            length = d.get_size()
            if isinstance(d, PartialObservableDeck):
                for i in range(length):
                    if not d.get_visibility_for_player(i, player) and lambda_func(d.get(i)):
                        all_cards.add(d.get(i))
            else:
                visibility_mode = d.get_visibility_mode()
                if visibility_mode == CoreConstants.VisibilityMode.VISIBLE_TO_ALL:
                    continue
                elif visibility_mode == CoreConstants.VisibilityMode.VISIBLE_TO_OWNER:
                    if d.get_owner_id() == player:
                        continue
                elif visibility_mode == CoreConstants.VisibilityMode.HIDDEN_TO_ALL:
                    for i in range(length):
                        if lambda_func(d.get(i)):
                            all_cards.add(d.get(i))
                elif visibility_mode == CoreConstants.VisibilityMode.TOP_VISIBLE_TO_ALL:
                    for i in range(1, length):
                        if lambda_func(d.get(i)):
                            all_cards.add(d.get(i))
                elif visibility_mode == CoreConstants.VisibilityMode.BOTTOM_VISIBLE_TO_ALL:
                    for i in range(length - 1):
                        if lambda_func(d.get(i)):
                            all_cards.add(d.get(i))
                elif visibility_mode == CoreConstants.VisibilityMode.MIXED_VISIBILITY:
                    raise AssertionError("Not supported: MIXED_VISIBILITY")

        all_cards.shuffle(rnd)

        # Put the shuffled cards back in place
        for d in decks:
            length = d.get_size()
            if isinstance(d, PartialObservableDeck):
                for i in range(length):
                    if not d.get_visibility_for_player(i, player) and lambda_func(d.get(i)):
                        d.set_component(i, all_cards.draw())
            else:
                visibility_mode = d.get_visibility_mode()
                if visibility_mode == CoreConstants.VisibilityMode.VISIBLE_TO_ALL:
                    continue
                elif visibility_mode == CoreConstants.VisibilityMode.VISIBLE_TO_OWNER:
                    if d.get_owner_id() == player:
                        continue
                elif visibility_mode == CoreConstants.VisibilityMode.HIDDEN_TO_ALL:
                    for i in range(length):
                        if lambda_func(d.get(i)):
                            d.set_component(i, all_cards.draw())
                elif visibility_mode == CoreConstants.VisibilityMode.TOP_VISIBLE_TO_ALL:
                    for i in range(1, length):
                        if lambda_func(d.get(i)):
                            d.set_component(i, all_cards.draw())
                elif visibility_mode == CoreConstants.VisibilityMode.BOTTOM_VISIBLE_TO_ALL:
                    for i in range(length - 1):
                        if lambda_func(d.get(i)):
                            d.set_component(i, all_cards.draw())
                elif visibility_mode == CoreConstants.VisibilityMode.MIXED_VISIBILITY:
                    raise AssertionError("Not supported: MIXED_VISIBILITY")
                
class PartialObservableDeck(Deck[T]):
    def __init__(self, deck_id: str, owner_id: int, default_visibility: Optional[List[bool]] = None, 
                 n_players: Optional[int] = None, visibility_mode: Optional[VisibilityMode] = None, 
                 component_id: Optional[int] = None):
        
        if default_visibility is not None:
            super().__init__(deck_id, owner_id, component_id, VisibilityMode.MIXED_VISIBILITY)
            self.deck_visibility = default_visibility
        elif n_players is not None and visibility_mode is not None:
            super().__init__(deck_id, owner_id, component_id, visibility_mode)
            self.deck_visibility = [False] * n_players
            if visibility_mode == VisibilityMode.VISIBLE_TO_ALL:
                self.deck_visibility = [True] * n_players
            elif visibility_mode == VisibilityMode.VISIBLE_TO_OWNER:
                self.deck_visibility[owner_id] = True
        else:
            raise ValueError("Either default_visibility or (n_players and visibility_mode) must be provided")
            
        self.element_visibility: List[List[bool]] = []

    def get_visibility_for_player(self, element_idx: int, player_id: int) -> bool:
        return self.element_visibility[element_idx][player_id]

    def get_visibility_of_component(self, element_idx: int) -> List[bool]:
        return self.element_visibility[element_idx]

    def get_visible_components(self, player_id: int) -> List[Optional[T]]:
        if not 0 <= player_id < len(self.deck_visibility):
            raise ValueError(f"playerID {player_id} needs to be in range [0, {len(self.deck_visibility)-1}]")
        
        visible_components = []
        for i in range(len(self.components)):
            if self.element_visibility[i][player_id]:
                visible_components.append(self.components[i])
            else:
                visible_components.append(None)
        return visible_components

    def is_component_visible(self, idx: int, player_id: int) -> bool:
        if not 0 <= player_id < len(self.deck_visibility):
            raise ValueError(f"playerID {player_id} needs to be in range [0, {len(self.deck_visibility)-1}]")
        return self.element_visibility[idx][player_id]

    def set_visibility(self, visibility_mode: VisibilityMode):
        super().set_visibility(visibility_mode)
        self.apply_visibility_mode()

    def set_components_with_visibility(self, components: List[T], visibility_per_player: List[List[bool]]):
        super().set_components(components)
        self.element_visibility = visibility_per_player

    def set_visibility_list(self, visibility: List[List[bool]]):
        for b in visibility:
            if len(b) != len(self.deck_visibility):
                raise ValueError(f"All entries of visibility need to have length {len(self.deck_visibility)}")
        self.element_visibility = visibility

    def apply_visibility_mode(self):
        if self.get_visibility_mode() == VisibilityMode.TOP_VISIBLE_TO_ALL:
            for j in range(len(self.deck_visibility)):
                self.element_visibility[0][j] = True
        if self.get_visibility_mode() == VisibilityMode.BOTTOM_VISIBLE_TO_ALL:
            for j in range(len(self.deck_visibility)):
                self.element_visibility[-1][j] = True

    def set_visibility_of_component(self, index: int, player_id: int, visibility: bool):
        if not 0 <= index < len(self.element_visibility):
            raise ValueError(f"component index {index} out of range")
        if not 0 <= player_id < len(self.deck_visibility):
            raise ValueError(f"playerID {player_id} out of range")
        self.element_visibility[index][player_id] = visibility

    def set_visibility_of_component_all(self, index: int, visibility: List[bool]):
        if not 0 <= index < len(self.element_visibility) or len(visibility) != len(self.deck_visibility):
            raise ValueError("Invalid index or visibility length")
        self.element_visibility[index] = visibility.copy()

    def add_with_visibility(self, component: T, visibility_per_player: List[bool]) -> bool:
        return self.add_with_visibility_at(component, 0, visibility_per_player)

    def add_with_visibility_at(self, component: T, index: int, visibility_per_player: List[bool]) -> bool:
        self.element_visibility.insert(index, visibility_per_player.copy())
        ret_value = super().add(component, index)
        self.apply_visibility_mode()
        return ret_value

    def add_deck(self, deck: Deck[T], index: int = 0) -> bool:
        if isinstance(deck, PartialObservableDeck):
            length = len(deck.components)
            for i in range(length):
                self.element_visibility.insert(index, deck.element_visibility[length - i - 1].copy())
        else:
            for _ in range(len(deck.components)):
                self.element_visibility.insert(index, self.deck_visibility.copy())
        
        ret_value = super().add_deck(deck, index)
        self.apply_visibility_mode()
        return ret_value

    def set_components(self, components: List[T]):
        super().set_components(components)
        self.element_visibility = [self.deck_visibility.copy() for _ in components]
        self.apply_visibility_mode()

    def pick(self, idx: int) -> Optional[T]:
        el = super().pick(idx)
        if el is not None:
            del self.element_visibility[idx]
        return el

    def add(self, component: T, index: int = 0) -> bool:
        return self.add_with_visibility_at(component, index, self.deck_visibility)

    def add_to_bottom(self, component: T) -> bool:
        if not self.components:
            return self.add_with_visibility_at(component, 0, self.deck_visibility)
        return self.add_with_visibility_at(component, len(self.components), self.deck_visibility)

    def remove(self, idx: int) -> bool:
        if super().remove(idx):
            del self.element_visibility[idx]
            return True
        return False

    def clear(self):
        super().clear()
        self.element_visibility.clear()

    def shuffle(self, rnd: random.Random):
        self.element_visibility = [self.deck_visibility.copy() for _ in self.components]
        super().shuffle(rnd)
        self.apply_visibility_mode()

    def shuffle_and_keep_visibility(self, rnd: random.Random):
        shuffled = self.shuffle_lists(self.components, self.element_visibility, rnd)
        self.components, self.element_visibility = shuffled.a, shuffled.b
        self.apply_visibility_mode()

    @staticmethod
    def shuffle_lists(comps: List[T], vis: List[List[bool]], rnd: random.Random) -> Pair[List[T], List[List[bool]]]:
        indices = list(range(len(comps)))
        rnd.shuffle(indices)
        new_comps = [comps[i] for i in indices]
        new_vis = [vis[i] for i in indices]
        return Pair(new_comps, new_vis)

    def redeterminise_unknown(self, rnd: random.Random, player_id: int):
        DeterminisationUtilities.reshuffle(player_id, [self], lambda c: True, rnd)

    def get_deck_visibility(self) -> List[bool]:
        return self.deck_visibility.copy()

    def copy(self) -> 'PartialObservableDeck[T]':
        new_deck = PartialObservableDeck(
            self.component_name, 
            self.owner_id, 
            self.deck_visibility.copy(), 
            component_id=self.component_id
        )
        self.copy_to(new_deck)
        return self.common_copy(new_deck)

    def copy_with_player(self, player_id: int) -> 'PartialObservableDeck[T]':
        new_deck = PartialObservableDeck(
            self.component_name, 
            self.owner_id, 
            self.deck_visibility.copy(), 
            component_id=self.component_id
        )
        self.copy_to(new_deck, player_id)
        return self.common_copy(new_deck)

    def common_copy(self, new_deck: 'PartialObservableDeck[T]') -> 'PartialObservableDeck[T]':
        new_deck.element_visibility = [v.copy() for v in self.element_visibility]
        return new_deck

    def to_string_with_player(self, gs: 'AbstractGameState', player_id: int) -> str:
        components_str = []
        for i in range(len(self.components)):
            if not self.is_component_visible(i, player_id) and gs.get_core_game_parameters().partial_observable:
                components_str.append("UNKNOWN")
            else:
                components_str.append(str(self.components[i]))
        return ",".join(components_str) if components_str else "EmptyDeck"

    def __str__(self) -> str:
        return super().__str__()
    

class Card(Component):
    
    def __init__(self, name=None, component_id=None):
        if name is None:
            super().__init__(CoreConstants.ComponentType.CARD)
        elif component_id is None:
            super().__init__(CoreConstants.ComponentType.CARD, name)
        else:
            super().__init__(CoreConstants.ComponentType.CARD, name, component_id)
    
    def copy(self):
        copy_card = Card(self.component_name, self.component_id)
        self.copy_component_to(copy_card)
        return copy_card
    
    def __str__(self):
        hash_name = self.get_property(CoreConstants.name_hash)
        if hash_name is not None and isinstance(hash_name, PropertyString):
            return hash_name.value
        return self.component_name
    
    def __hash__(self):
        return super().__hash__()