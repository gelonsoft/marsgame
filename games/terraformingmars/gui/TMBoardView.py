import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from core.components import Counter
from games.terraformingmars import TMGameState, TMTypes
from games.terraformingmars.components import TMMapTile

class TMBoardView:
    def __init__(self, gui, gs: TMGameState):
        self.gs = gs
        self.gui = gui
        self.highlight = []
        self.rects = {}
        
        self.player_colors = [
            (255, 0, 0),    # Red
            (255, 255, 0),  # Yellow
            (0, 0, 255),    # Blue
            (0, 255, 0)     # Green
        ]
        
        self.default_item_size = 30
        self.offset_x = 10
        self.spacing = 10
        self.map_tile_bg = (97, 97, 97, 70)
    
    def draw(self):
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Draw global parameters
        draw_order = TMTypes.GlobalParameter.getDrawOrder(self.gs.getGameParameters())
        for i, param in enumerate(draw_order):
            self._draw_counter(ax, param, self.gs.getGlobalParameters().get(param), 
                              self.offset_x + i * self.default_item_size * 2, 0)
        
        # Draw grid board
        grid_board = self.gs.getBoard()
        board_x = self.offset_x + (len(self.gs.getGlobalParameters()) + 1) * self.default_item_size + 10
        board_y = self.default_item_size
        
        for i in range(grid_board.getHeight()):
            for j in range(grid_board.getWidth()):
                offset_x = self.default_item_size / 2
                if i % 2 == 1:
                    offset_x += self.default_item_size / 2
                
                x = board_x + offset_x + j * self.default_item_size
                y = board_y + self.default_item_size / 2 + i * self.default_item_size
                
                self._draw_cell(ax, grid_board.getElement(j, i), x, y)
                rect = Rectangle((x - self.default_item_size/2, y - self.default_item_size/2), 
                                self.default_item_size, self.default_item_size)
                self.rects[rect] = f"grid-{j}-{i}"
        
        # Draw TR for all players
        tr_y = board_y + grid_board.getHeight() * self.default_item_size + self.spacing * 2
        for i in range(self.gs.getNPlayers()):
            tr_value = self.gs.getPlayerResources()[i].get(TMTypes.Resource.TR).getValue()
            ax.text(self.offset_x + i * self.default_item_size * 3 + self.default_item_size * 1.5,
                    tr_y + self.default_item_size / 2,
                    f"p{i}: {tr_value}", 
                    color=self._to_rgb(self.player_colors[i]),
                    ha='center', va='center')
        
        # Draw highlights
        if self.highlight:
            for rect in self.highlight:
                highlight_rect = Rectangle(rect.get_xy(), rect.get_width(), rect.get_height(),
                                         linewidth=3, edgecolor='green', facecolor='none')
                ax.add_patch(highlight_rect)
        
        st.pyplot(fig)
    
    def _draw_cell(self, ax, element: TMMapTile, x, y):
        if element is None:
            return
            
        # Create hexagon
        hex_coords = []
        for i in range(6):
            hex_coords.append((
                x + self.default_item_size/2 * np.cos(np.pi/2 + i * 2 * np.pi / 6),
                y + self.default_item_size/2 * np.sin(np.pi/2 + i * 2 * np.pi / 6)
            ))
        
        # Draw hexagon
        hex_patch = Polygon(hex_coords, closed=True, 
                           facecolor=self.map_tile_bg,
                           edgecolor=self._to_rgb(element.getTileType().getOutlineColor()))
        ax.add_patch(hex_patch)
        
        # Draw tile if placed
        if element.getTilePlaced():
            # Draw tile image (placeholder)
            tile_rect = Rectangle((x - self.default_item_size/4, y - self.default_item_size/4),
                                 self.default_item_size/2, self.default_item_size/2,
                                 facecolor='gray')
            ax.add_patch(tile_rect)
            
            # Draw owner indicator
            if element.getOwnerId() >= 0:
                owner_rect = Rectangle((x - self.default_item_size/6, y - self.default_item_size/6),
                                      self.default_item_size/3, self.default_item_size/3,
                                      facecolor=self._to_rgb(self.player_colors[element.getOwnerId()]),
                                      edgecolor='black')
                ax.add_patch(owner_rect)
    
    def _draw_counter(self, ax, param: TMTypes.GlobalParameter, counter: Counter, x, y):
        if param == TMTypes.GlobalParameter.OceanTiles:
            # Draw ocean tile counter
            ocean_rect = Rectangle((x, y), self.default_item_size, self.default_item_size,
                                  facecolor='blue', edgecolor='black')
            ax.add_patch(ocean_rect)
            
            ax.text(x + self.default_item_size/2, y + self.default_item_size/2,
                   f"{counter.getValue()}/{counter.getMaximum()}",
                   ha='center', va='center', color='yellow')
        else:
            # Draw parameter track
            steps = counter.getValues().length
            track_height = steps * self.default_item_size/2
            
            # Draw track background
            track_rect = Rectangle((x, y), self.default_item_size, track_height,
                                  facecolor='lightgray', edgecolor='black')
            ax.add_patch(track_rect)
            
            # Draw current level
            current_level = counter.getValueIdx()
            if current_level >= 0:
                level_rect = Rectangle((x, y + track_height - (current_level + 1) * self.default_item_size/2),
                                      self.default_item_size, self.default_item_size/2,
                                      facecolor=self._to_rgb(param.getColor()))
                ax.add_patch(level_rect)
            
            # Draw values
            for i, val in enumerate(counter.getValues()):
                ax.text(x + self.default_item_size/2, 
                       y + track_height - (i + 0.5) * self.default_item_size/2,
                       str(val), ha='center', va='center')
    
    def _to_rgb(self, color):
        return (color.getRed()/255, color.getGreen()/255, color.getBlue()/255)
    
    def update(self, gs: TMGameState):
        self.gs = gs
    
    def clearHighlights(self):
        self.highlight = []