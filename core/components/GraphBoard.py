from .Component import Component,IComponentContainer
from .BoardNode import BoardNode
from core import CoreConstants

from core.properties import Property, PropertyString, PropertyStringArray
import json
from typing import List, Dict, Collection, Optional, Any, Union
import utilities.Hash as Hash

class GraphBoard(Component, IComponentContainer[BoardNode]):
    
    def __init__(self, name: str = None, ID: int = None):
        if name is not None and ID is not None:
            super().__init__(CoreConstants.ComponentType.BOARD, name, ID)
        elif name is not None:
            super().__init__(CoreConstants.ComponentType.BOARD, name)
        elif ID is not None:
            super().__init__(CoreConstants.ComponentType.BOARD, ID)
        else:
            super().__init__(CoreConstants.ComponentType.BOARD)
        self.board_nodes: Dict[int, BoardNode] = {}

    def copy(self) -> 'GraphBoard':
        b = GraphBoard(self.component_name, self.component_id)
        node_copies: Dict[int, BoardNode] = {}
        
        # Copy board nodes
        for bn in self.board_nodes.values():
            bn_copy = BoardNode(bn.max_neighbours, "", bn.component_id)
            bn.copy_component_to(bn_copy)
            node_copies[bn.component_id] = bn_copy
        
        # Assign neighbours
        for bn in self.board_nodes.values():
            bn_copy = node_copies[bn.component_id]
            for neighbour in bn.get_neighbours().keys():
                bn_copy.add_neighbour_with_cost(node_copies[neighbour.component_id])
            for neighbour, side in bn.get_neighbour_side_mapping().items():
                bn_copy.add_neighbour_with_cost(node_copies[neighbour.component_id], side)
        
        # Assign new neighbours
        b.set_board_nodes(list(node_copies.values()))
        # Copy properties
        self.copy_component_to(b)
        return b

    def get_node_by_property(self, prop_id: int, p: Property) -> Optional[BoardNode]:
        for n in self.board_nodes.values():
            prop = n.get_property(prop_id)
            if prop is not None and prop.equals(p):
                return n
        return None

    def get_node_by_string_property(self, prop_id: int, value: str) -> Optional[BoardNode]:
        return self.get_node_by_property(prop_id, PropertyString(value))

    def get_board_nodes(self) -> Collection[BoardNode]:
        return self.board_nodes.values()

    def get_node_by_id(self, id: int) -> Optional[BoardNode]:
        return self.board_nodes.get(id)

    def set_board_nodes(self, board_nodes: Union[List[BoardNode], Dict[int, BoardNode]]) -> None:
        if isinstance(board_nodes, list):
            for bn in board_nodes:
                self.board_nodes[bn.component_id] = bn
        else:
            self.board_nodes = board_nodes

    def add_board_node(self, bn: BoardNode) -> None:
        self.board_nodes[bn.component_id] = bn

    def remove_board_node(self, bn: BoardNode) -> None:
        self.board_nodes.pop(bn.component_id, None)

    def break_connection(self, gs: Any, bn1: BoardNode, bn2: BoardNode) -> None:
        bn1.remove_neighbour(bn2)
        bn2.remove_neighbour(bn1)

        # Check if they have at least 1 more neighbour on this board. If not, remove node from this board
        in_board = any(n.component_id in self.board_nodes for n in bn1.get_neighbours().keys())
        if not in_board:
            self.board_nodes.pop(bn1.component_id, None)

        in_board = any(n.component_id in self.board_nodes for n in bn2.get_neighbours().keys())
        if not in_board:
            self.board_nodes.pop(bn2.component_id, None)

    def add_connection(self, bn1: BoardNode, bn2: BoardNode, edge_value: Optional[int] = None) -> None:
        if edge_value is not None:
            bn1.add_neighbour_with_cost(bn2, edge_value)
            bn2.add_neighbour_with_cost(bn1, edge_value)
        else:
            bn1.add_neighbour_with_cost(bn2)
            bn2.add_neighbour_with_cost(bn1)
        
        if bn1.component_id not in self.board_nodes:
            self.board_nodes[bn1.component_id] = bn1
        if bn2.component_id not in self.board_nodes:
            self.board_nodes[bn2.component_id] = bn2

    def add_connection_by_id(self, bn1_id: int, bn2_id: int, edge_value: Optional[int] = None) -> None:
        bn1 = self.board_nodes.get(bn1_id)
        bn2 = self.board_nodes.get(bn2_id)
        if bn1 and bn2:
            self.add_connection(bn1, bn2, edge_value)

    @staticmethod
    def load_boards(filename: str) -> List['GraphBoard']:
        graph_boards = []
        try:
            with open(filename, 'r') as file:
                data = json.load(file)
                for item in data:
                    new_graph_board = GraphBoard()
                    new_graph_board.load_board(item)
                    graph_boards.append(new_graph_board)
        except (IOError, json.JSONDecodeError) as e:
            print(e)
        return graph_boards

    def load_board(self, board: Dict[str, Any]) -> None:
        self.component_name = board.get("id")
        board_type = board.get("type")
        vertices_key = board.get("verticesKey")
        neighbours_key = board.get("neighboursKey")
        max_neighbours = int(board.get("maxNeighbours"))

        self.properties[Hash.get_instance().hash("boardType")] = PropertyString("boardType", board_type)
        if board.get("img") is not None:
            self.properties[CoreConstants.img_hash] = PropertyString("img", board.get("img"))

        node_list = board.get("nodes", [])
        for node_data in node_list:
            new_bn = BoardNode()
            new_bn.load_board_node(node_data)
            new_bn.set_component_name(new_bn.get_property(CoreConstants.name_hash).value)
            new_bn.set_max_neighbours(max_neighbours)
            self.board_nodes[new_bn.component_id] = new_bn

        hash_neighbours = Hash.get_instance().hash(neighbours_key)
        hash_vertices = Hash.get_instance().hash(vertices_key)

        for bn in self.board_nodes.values():
            p = bn.get_property(hash_neighbours)
            if isinstance(p, PropertyStringArray):
                for s in p.values:
                    neigh = self.get_node_by_property(hash_vertices, PropertyString(s))
                    if neigh is not None:
                        bn.add_neighbour_with_cost(neigh)
                        neigh.add_neighbour_with_cost(bn)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, GraphBoard):
            return self.component_id == other.component_id and self.board_nodes == other.board_nodes
        return False

    def __hash__(self) -> int:
        return hash((self.component_id, frozenset(self.board_nodes.items())))

    def get_components(self) -> List[BoardNode]:
        return list(self.board_nodes.values())

    def get_board_node_map(self) -> Dict[int, BoardNode]:
        return self.board_nodes

    def get_visibility_mode(self) -> CoreConstants.VisibilityMode:
        return CoreConstants.VisibilityMode.VISIBLE_TO_ALL