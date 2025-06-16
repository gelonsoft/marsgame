from typing import List, Dict, Tuple, Optional, Any, Union
import json
from copy import deepcopy
from core.components import BoardNode, GraphBoard
from .Component import Component,IComponentContainer
from core.properties import PropertyString, PropertyVector2D
from core import CoreConstants
from utilities import Pair, Vector2D, Utils


class GridBoard(Component, IComponentContainer):
    
    def __init__(self, *args):
        """
        Multiple constructors handled through argument checking:
        - GridBoard() - empty
        - GridBoard(width, height)
        - GridBoard(width, height, default_value)
        - GridBoard(grid)
        - GridBoard(grid, ID)
        - GridBoard(width, height, ID)
        - GridBoard(original) - copy constructor
        """
        super().__init__(CoreConstants.ComponentType.BOARD)
        
        if len(args) == 0:
            # Empty constructor
            self.width = 0
            self.height = 0
            self.grid = []
        elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int):
            # Width, height constructor
            self.width = args[0]
            self.height = args[1]
            self.grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        elif len(args) == 3 and isinstance(args[0], int) and isinstance(args[1], int) and isinstance(args[2], BoardNode):
            # Width, height, default value constructor
            self.width = args[0]
            self.height = args[1]
            self.grid = [[deepcopy(args[2]) for _ in range(self.width)] for _ in range(self.height)]
        elif len(args) == 1 and isinstance(args[0], list) and all(isinstance(row, list) for row in args[0]):
            # Grid constructor
            self.width = len(args[0][0])
            self.height = len(args[0])
            self.grid = deepcopy(args[0])
        elif len(args) == 2 and isinstance(args[0], list) and all(isinstance(row, list) for row in args[0]) and isinstance(args[1], int):
            # Grid with ID constructor
            super().__init__(CoreConstants.ComponentType.BOARD, args[1])
            self.width = len(args[0][0])
            self.height = len(args[0])
            self.grid = deepcopy(args[0])
        elif len(args) == 2 and isinstance(args[0], int) and isinstance(args[1], int) and isinstance(args[2], int):
            # Width, height, ID constructor
            super().__init__(CoreConstants.ComponentType.BOARD, args[2])
            self.width = args[0]
            self.height = args[1]
            self.grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        elif len(args) == 1 and isinstance(args[0], GridBoard):
            # Copy constructor
            orig = args[0]
            super().__init__(CoreConstants.ComponentType.BOARD)
            self.width = orig.width
            self.height = orig.height
            self.grid = deepcopy(orig.grid)
        else:
            raise ValueError("Invalid constructor arguments")

    def get_width(self) -> int:
        return self.width

    def get_height(self) -> int:
        return self.height

    def set_width(self, width: int, offset: int = 0) -> None:
        self.set_width_height(width, self.height, offset, 0)

    def set_height(self, height: int, offset: int = 0) -> None:
        self.set_width_height(self.width, height, 0, offset)

    def set_width_height(self, width: int, height: int, offset_x: int = 0, offset_y: int = 0) -> None:
        if offset_x + self.width > width:
            offset_x = 0
        if offset_y + self.height > height:
            offset_y = 0

        w = min(width, self.width)
        h = min(height, self.height)

        self.width = width
        self.height = height

        new_grid = [[None for _ in range(width)] for _ in range(height)]
        for i in range(h):
            for j in range(w):
                if i + offset_y < height and j + offset_x < width:
                    new_grid[i + offset_y][j + offset_x] = self.grid[i][j]
        self.grid = new_grid

    def set_element(self, x: int, y: int, value: BoardNode) -> bool:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = value
            return True
        return False

    def set_element_vector(self, pos: Vector2D, value: BoardNode) -> bool:
        return self.set_element(pos.x, pos.y, value)

    def get_element(self, x: int, y: int) -> Optional[BoardNode]:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return None

    def get_element_vector(self, pos: Vector2D) -> Optional[BoardNode]:
        return self.get_element(pos.x, pos.y)

    def get_grid_values(self) -> List[List[BoardNode]]:
        return self.grid

    def get_empty_cells(self, default_element: BoardNode) -> List[Vector2D]:
        empty_cells = []
        for y in range(self.height):
            for x in range(self.width):
                element = self.get_element(x, y)
                if element is None or element == default_element:
                    empty_cells.append(Vector2D(x, y))
        return empty_cells

    def rotate(self, orientation: int) -> List[List[BoardNode]]:
        copy = self.copy()
        orientation %= 4
        for _ in range(orientation):
            copy.grid = self.rotate_clockwise(copy.grid)
        return copy.grid

    @staticmethod
    def rotate_clockwise(original: List[List[BoardNode]]) -> List[List[BoardNode]]:
        m = len(original)
        n = len(original[0]) if m > 0 else 0
        grid = [[None for _ in range(m)] for _ in range(n)]
        for r in range(m):
            for c in range(n):
                grid[c][m - 1 - r] = original[r][c]
        return grid

    def flatten_grid(self) -> List[BoardNode]:
        return [item for sublist in self.grid for item in sublist]

    def copy(self) -> 'GridBoard':
        grid_copy = [[None for _ in range(self.width)] for _ in range(self.height)]
        node_copies = {}
        
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] is not None:
                    grid_copy[y][x] = BoardNode(self.grid[y][x])
                    node_copies[grid_copy[y][x].component_id] = grid_copy[y][x]
        
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] is not None:
                    for neighbour, cost in self.grid[y][x].get_neighbours().items():
                        grid_copy[y][x].add_neighbour_with_cost(node_copies[neighbour.component_id], cost)
                    for neighbour, side in self.grid[y][x].get_neighbour_side_mapping().items():
                        grid_copy[y][x].add_neighbour_on_side(node_copies[neighbour.component_id], side)
        
        g = GridBoard(grid_copy, self.component_id)
        self.copy_component_to(g)
        return g

    def copy_new_id(self) -> 'GridBoard':
        grid_copy = [[None for _ in range(self.width)] for _ in range(self.height)]
        node_copies = {}
        
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] is not None:
                    grid_copy[y][x] = BoardNode(self.grid[y][x])
                    node_copies[grid_copy[y][x].component_id] = grid_copy[y][x]
        
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] is not None:
                    for neighbour, cost in self.grid[y][x].get_neighbours().items():
                        grid_copy[y][x].add_neighbour_with_cost(node_copies[neighbour.component_id], cost)
                    for neighbour, side in self.grid[y][x].get_neighbour_side_mapping().items():
                        grid_copy[y][x].add_neighbour_on_side(node_copies[neighbour.component_id], side)
        
        g = GridBoard(grid_copy)
        self.copy_component_to(g)
        return g

    def empty_copy(self) -> 'GridBoard':
        g = GridBoard(self.width, self.height, self.component_id)
        self.copy_component_to(g)
        return g

    def __str__(self) -> str:
        return "\n".join(
            " ".join(
                cell.get_component_name() if cell is not None else "None"
                for cell in row
            )
            for row in self.grid
        )

    @staticmethod
    def load_boards(filename: str) -> List['GridBoard']:
        grid_boards = []
        with open(filename, 'r') as f:
            data = json.load(f)
            for board_data in data:
                new_grid_board = GridBoard()
                new_grid_board.load_board(board_data)
                grid_boards.append(new_grid_board)
        return grid_boards

    def load_board(self, board: Dict[str, Any]) -> None:
        self.component_name = board.get("id", "")
        
        size = board.get("size", [0, 0])
        self.width = size[0]
        self.height = size[1]
        
        if "img" in board:
            self.properties[CoreConstants.img_hash] = PropertyString(board["img"])
        
        self.grid = [[None for _ in range(self.width)] for _ in range(self.height)]
        
        grids = board.get("grid", [])
        y = 0
        for g in grids:
            if isinstance(g[0], list):
                y = 0
                for row in g:
                    x = 0
                    for o1 in row:
                        if isinstance(o1, str):
                            bn = BoardNode(-1, o1)
                            self.set_element(x, y, bn)
                        else:
                            self.set_element(x, y, o1)
                        x += 1
                    y += 1
            else:
                x = 0
                for o1 in g:
                    if isinstance(o1, str):
                        bn = BoardNode(-1, o1)
                        self.set_element(x, y, bn)
                    else:
                        self.set_element(x, y, o1)
                    x += 1
                y += 1

    def to_graph_board(self, way8: bool) -> 'GraphBoard':
        gb = GraphBoard(self.component_name, self.component_id)
        bn_mapping = {}
        
        for y in range(self.height):
            for x in range(self.width):
                element = self.get_element(x, y)
                if element is not None:
                    bn = BoardNode(-1, element.get_component_name())
                    bn.set_property(PropertyVector2D("coordinates", Vector2D(x, y)))
                    bn.set_property(PropertyString("terrain", element.get_component_name()))
                    gb.add_board_node(bn)
                    bn_mapping[Vector2D(x, y)] = bn
        
        for y in range(self.height):
            for x in range(self.width):
                bn = bn_mapping.get(Vector2D(x, y))
                if bn is not None:
                    neighbours = Utils.get_neighbourhood(x, y, self.width, self.height, way8)
                    for neighbour in neighbours:
                        bn2 = bn_mapping.get(neighbour)
                        if bn2 is not None:
                            gb.add_connection(bn, bn2)
        return gb

    def to_graph_board_with_neighbours(self, neighbours: List[Pair[Vector2D, Vector2D]]) -> 'GraphBoard':
        gb = GraphBoard(self.component_name, self.component_id)
        bn_mapping = {}
        
        for y in range(self.height):
            for x in range(self.width):
                element = self.get_element(x, y)
                if element is not None:
                    bn = BoardNode(-1, element.get_component_name())
                    bn.set_property(PropertyVector2D("coordinates", Vector2D(x, y)))
                    bn.set_property(PropertyString("terrain", element.get_component_name()))
                    gb.add_board_node(bn)
                    bn_mapping[Vector2D(x, y)] = bn
        
        for p in neighbours:
            bn1 = bn_mapping.get(p.a)
            bn2 = bn_mapping.get(p.b)
            if bn1 is not None and bn2 is not None:
                gb.add_connection(bn1, bn2)
        return gb

    def set_neighbours(self, neighbours: List[Pair[Vector2D, Vector2D]]) -> None:
        bn_mapping = {}
        
        for y in range(self.height):
            for x in range(self.width):
                bn = self.get_element(x, y)
                if bn is not None:
                    bn.set_property(PropertyVector2D("coordinates", Vector2D(x, y)))
                    bn_mapping[Vector2D(x, y)] = bn
        
        for p in neighbours:
            bn1 = bn_mapping.get(p.a)
            bn2 = bn_mapping.get(p.b)
            if bn1 is not None and bn2 is not None:
                bn1.add_neighbour_with_cost(bn2)
                bn2.add_neighbour_with_cost(bn1)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridBoard):
            return False
        return (self.component_id == other.component_id and 
                self.flatten_grid() == other.flatten_grid())

    def __hash__(self) -> int:
        return hash(self.component_id) + 5 * hash(tuple(self.flatten_grid()))

    def get_components(self) -> List[BoardNode]:
        return self.flatten_grid()

    def get_visibility_mode(self) -> CoreConstants.VisibilityMode:
        return CoreConstants.VisibilityMode.VISIBLE_TO_ALL