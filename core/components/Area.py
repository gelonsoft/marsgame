from typing import Dict, List, Optional, TypeVar, Any

from core.CoreConstants import VisibilityMode
from .Component import Component, ComponentType, IComponentContainer, Deck


class Area(Component):
    def __init__(self, owner: int, name: str, component_id: Optional[int] = None):
        super().__init__(ComponentType.AREA, name, component_id)
        self.components: Dict[int, Component] = {}
        self.owner_id = owner

    def copy(self) -> 'Area':
        new_area = Area(self.owner_id, self.component_name, self.component_id)
        new_area.components = {k: v.copy() for k, v in self.components.items()}
        self.copy_component_to(new_area)
        return new_area

    def empty_copy(self) -> 'Area':
        return Area(self.owner_id, self.component_name, self.component_id)

    def clear(self):
        self.components.clear()

    def get_components_map(self) -> Dict[int, Component]:
        return self.components

    def get_visibility_mode(self) -> VisibilityMode:
        return VisibilityMode.VISIBLE_TO_ALL

    def get_components(self) -> List[Component]:
        return list(self.components.values())

    def get_component(self, key: int) -> Optional[Component]:
        return self.components.get(key)

    def put_component(self, key: int, component: Component):
        self.components[key] = component

    def put_component_by_id(self, component: Component):
        if component is None:
            return
        self.components[component.component_id] = component
        if isinstance(component, IComponentContainer):
            for nested_c in component.get_components():
                if nested_c is not None:
                    self.put_component_by_id(nested_c)

    def remove_component(self, component: Component):
        if isinstance(component, (Deck, Area)):
            raise ValueError("Not yet implemented for Decks or Areas")
        if component.component_id in self.components:
            del self.components[component.component_id]
        else:
            raise ValueError(f"Cannot remove Component as it is not here: {component.component_id}")

    def put_components(self, components: List[Component]):
        for c in components:
            self.put_component_by_id(c)

    def size(self) -> int:
        return len(self.components)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Area):
            return (self.component_id == other.component_id and 
                    self.components == other.components)
        return False

    def __hash__(self) -> int:
        return hash((self.component_id, frozenset(self.components.items())))