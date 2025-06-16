from core.components import Component
from core import CoreConstants
from typing import Dict, Optional
import json

class BoardNode(Component):
    default_cost = 1.0

    def __init__(self, max_neighbours: int = -1, name: str = "", component_id: Optional[int] = None):
        """
        Initialize a BoardNode.
        
        Args:
            max_neighbours: Maximum number of neighbours for this node. -1 means no limit.
            name: Name of the node.
            component_id: Optional ID for the component. If None, an ID will be generated.
        """
        if component_id is not None:
            super().__init__(CoreConstants.ComponentType.BOARD_NODE, name, component_id)
        else:
            super().__init__(CoreConstants.ComponentType.BOARD_NODE, name)
        self.max_neighbours = max_neighbours
        self.neighbours: Dict['BoardNode', float] = {}  # Neighbours and their traversal costs
        self.neighbour_side_mapping: Dict['BoardNode', int] = {}  # Neighbours mapped to their side indices

    @classmethod
    def from_existing(cls, other: 'BoardNode') -> 'BoardNode':
        """
        Create a new BoardNode as a copy of an existing one.
        
        Args:
            other: The BoardNode to copy.
            
        Returns:
            A new BoardNode instance with the same properties as `other`.
        """
        new_node = cls(other.max_neighbours, other.component_name, other.component_id)
        other.copy_component_to(new_node)
        return new_node

    def add_neighbour_with_cost(self, neighbour: 'BoardNode', cost: float = default_cost) -> None:
        """
        Add a neighbour to this node with an associated traversal cost.
        
        Args:
            neighbour: The neighbour node to add.
            cost: The cost to traverse to this neighbour. Defaults to `default_cost`.
        """
        if self.max_neighbours == -1 or len(self.neighbours) < self.max_neighbours:
            self.neighbours[neighbour] = cost

    def remove_neighbour(self, neighbour: 'BoardNode') -> bool:
        """
        Remove a neighbour from this node.
        
        Args:
            neighbour: The neighbour node to remove.
            
        Returns:
            True if the neighbour was removed successfully, False otherwise.
        """
        if neighbour in self.neighbours:
            del self.neighbours[neighbour]
            if neighbour in self.neighbour_side_mapping:
                del self.neighbour_side_mapping[neighbour]
            return True
        return False

    def add_neighbour_on_side_with_cost(self, neighbour: 'BoardNode', side: int, cost: float = default_cost) -> bool:
        """
        Add a neighbour to a specific side of this node with an associated traversal cost.
        
        Args:
            neighbour: The neighbour node to add.
            side: The side index of this node where the neighbour is added.
            cost: The cost to traverse to this neighbour. Defaults to `default_cost`.
            
        Returns:
            True if the neighbour was added successfully, False otherwise.
        """
        if (self.max_neighbours == -1 or 
            (len(self.neighbours) < self.max_neighbours and side <= self.max_neighbours)):
            if neighbour not in self.neighbours and neighbour not in self.neighbour_side_mapping:
                self.neighbours[neighbour] = cost
                self.neighbour_side_mapping[neighbour] = side
                return True
        return False

    def add_neighbour_on_side(self, neighbour: 'BoardNode', side: int) -> bool:
        """
        Add a neighbour to a specific side of this node with the default traversal cost.
        
        Args:
            neighbour: The neighbour node to add.
            side: The side index of this node where the neighbour is added.
            
        Returns:
            True if the neighbour was added successfully, False otherwise.
        """
        return self.add_neighbour_on_side_with_cost(neighbour, side, self.default_cost)

    def copy(self) -> 'BoardNode':
        """
        Create a copy of this node.
        
        Returns:
            A new BoardNode instance with the same properties as this node.
        """
        return BoardNode.from_existing(self)

    def copy_component_to(self, target: 'Component') -> None:
        """
        Copy the properties of this component to another component.
        
        Args:
            target: The target component to copy properties to.
            
        Raises:
            RuntimeError: If the target is not a BoardNode.
        """
        if not isinstance(target, BoardNode):
            raise RuntimeError("BoardNode.copy_component_to(): Trying to copy to an incompatible component type")
        super().copy_component_to(target)

    def get_neighbours(self) -> Dict['BoardNode', float]:
        """
        Get all neighbours of this node and their traversal costs.
        
        Returns:
            A dictionary mapping neighbour nodes to their traversal costs.
        """
        return self.neighbours

    def clear_neighbours(self) -> None:
        """
        Clear all neighbours and side mappings of this node.
        """
        self.neighbours.clear()
        self.neighbour_side_mapping.clear()

    def get_neighbour_cost(self, neighbour: 'BoardNode') -> float:
        """
        Get the traversal cost to a specific neighbour.
        
        Args:
            neighbour: The neighbour node to query.
            
        Returns:
            The traversal cost to the neighbour.
            
        Raises:
            RuntimeError: If the neighbour is not found.
        """
        if neighbour in self.neighbours:
            return self.neighbours[neighbour]
        raise RuntimeError("BoardNode.get_neighbour_cost(): Accessing cost of a non-neighbour")

    def get_neighbour_side_mapping(self) -> Dict['BoardNode', int]:
        """
        Get the side mappings of all neighbours.
        
        Returns:
            A dictionary mapping neighbour nodes to their side indices.
        """
        return self.neighbour_side_mapping

    def get_max_neighbours(self) -> int:
        """
        Get the maximum number of neighbours allowed for this node.
        
        Returns:
            The maximum number of neighbours, or -1 if there is no limit.
        """
        return self.max_neighbours

    def set_max_neighbours(self, max_neighbours: int) -> None:
        """
        Set the maximum number of neighbours allowed for this node.
        
        Args:
            max_neighbours: The new maximum number of neighbours. Use -1 for no limit.
        """
        self.max_neighbours = max_neighbours

    def load_board_node(self, node_data: dict) -> None:
        """
        Load node data from a JSON-like dictionary.
        
        Args:
            node_data: A dictionary containing the node data.
        """
        self.component_name = node_data["name"][1]
        self.parse_component(self, node_data)

    def __str__(self) -> str:
        """
        Get a string representation of this node.
        
        Returns:
            A string describing the node's ID, maximum neighbours, and properties.
        """
        props_str = "; ".join(f"{prop.hash_string()}: {prop}" for prop in self.properties.values())
        return f"{{id: {self.component_id}; max_neighbours: {self.max_neighbours}; {props_str}}}"

    def __hash__(self) -> int:
        """
        Get the hash value of this node, which is its component ID.
        
        Returns:
            The component ID as the hash value.
        """
        return self.component_id