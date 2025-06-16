import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from games.terraformingmars import TMGameState
from evaluation.summarisers import TAGOccurrenceStatSummary

class TMBoardHeatMap:
    def __init__(self, gs: TMGameState, stats: TAGOccurrenceStatSummary, n_games: int):
        self.gs = gs
        self.stats = stats
        self.n_games = n_games
        
        # Calculate min/max counts
        self.min_count = min(stats.getElements().values()) if stats.getElements() else 0
        self.max_count = max(stats.getElements().values()) if stats.getElements() else 0
        
        self.tile_size = 30
        self.spacing = 10
        self.map_tile_bg = (97, 97, 97, 70)
        
    def draw(self):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        ax.axis('off')
        
        grid_board = self.gs.getBoard()
        
        # Draw hex grid
        for i in range(grid_board.getHeight()):
            for j in range(grid_board.getWidth()):
                offset_x = self.tile_size / 2
                if i % 2 == 1:
                    offset_x += self.tile_size / 2
                
                x = offset_x + j * self.tile_size
                y = self.tile_size + i * self.tile_size
                
                self._draw_hex(ax, x, y, (j, i))
        
        # Draw extra tiles
        y = self.tile_size * (grid_board.getHeight() + 1)
        x = max(0, (grid_board.getWidth() * self.tile_size) / 2 - 
                len(self.gs.getExtraTiles()) * self.tile_size / 2)
        
        for i, mt in enumerate(self.gs.getExtraTiles()):
            self._draw_hex(ax, x, y, mt.getComponentName())
            x += self.tile_size * 2
            
            if x + self.tile_size > grid_board.getWidth() * self.tile_size:
                x = max(0, (grid_board.getWidth() * self.tile_size) / 2 - 
                        (len(self.gs.getExtraTiles()) - i - 1) * self.tile_size / 2 + 
                        self.tile_size / 2)
                y += self.tile_size
        
        st.pyplot(fig)
    
    def _draw_hex(self, ax, x, y, identifier):
        # Create hexagon coordinates
        hex_coords = []
        for k in range(6):
            hex_coords.append((
                x + self.tile_size / 2 * np.cos(np.pi/2 + k * 2 * np.pi / 6),
                y + self.tile_size / 2 * np.sin(np.pi/2 + k * 2 * np.pi / 6)
            ))
        
        # Draw hexagon
        hex_patch = Polygon(hex_coords, closed=True, 
                           facecolor=(self.map_tile_bg), 
                           edgecolor='black')
        ax.add_patch(hex_patch)
        
        # Add heatmap if stats exist
        if isinstance(identifier, tuple):
            search_key = f"({identifier[0]}-{identifier[1]})"
        else:
            search_key = identifier
            
        if self.stats.getElements().get(search_key):
            count = self.stats.getElements()[search_key]
            perc = count / self.n_games
            alpha = 0.3 + perc * 0.7
            heat_color = (161/255, 64/255, 245/255, alpha)
            
            heat_patch = Polygon(hex_coords, closed=True, facecolor=heat_color)
            ax.add_patch(heat_patch)
            
            # Add count text
            ax.text(x, y, str(count), ha='center', va='center', fontsize=8)