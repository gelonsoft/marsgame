import streamlit as st
from typing import List
from games.terraformingmars import TMGameState, TMTypes
from games.terraformingmars.components import Award, Milestone

class TMPlayerView:
    def __init__(self, gs: TMGameState, player: int):
        self.gs = gs
        self.player = player
        self.highlight = []
        
        self.player_colors = [
            (255, 0, 0),    # Red
            (255, 255, 0),  # Yellow
            (0, 0, 255),    # Blue
            (0, 255, 0)     # Green
        ]
        
        self.default_item_size = 30
        self.offset_x = 10
        self.spacing = 10
    
    def display(self):
        st.header(f"Player {self.player} Resources")
        
        # Display resources and production
        cols = st.columns(len([r for r in TMTypes.Resource.values() if r.isPlayerBoardRes()]))
        col_idx = 0
        for res in TMTypes.Resource.values():
            if not res.isPlayerBoardRes():
                continue
                
            with cols[col_idx]:
                st.write(f"**{res.name()}**")
                st.write(f"Amount: {self.gs.getPlayerResources()[self.player].get(res).getValue()}")
                st.write(f"Production: {self.gs.getPlayerProduction()[self.player].get(res).getValue()}")
            col_idx += 1
        
        # Display tags
        st.subheader("Tags Played")
        tags_cols = st.columns(len(TMTypes.Tag.values()))
        for i, tag in enumerate(TMTypes.Tag.values()):
            with tags_cols[i]:
                st.write(f"**{tag.name()}**")
                st.write(f"{self.gs.getPlayerCardsPlayedTags()[self.player].get(tag).getValue()}")
        
        # Display card types
        st.subheader("Card Types Played")
        for card_type in TMTypes.CardType.values():
            if card_type.isPlayableStandard():
                count = self.gs.getPlayerCardsPlayedTypes()[self.player].get(card_type).getValue()
                st.write(f"{card_type.name()}: {count}")
        
        # Display tiles placed
        st.subheader("Tiles Placed")
        tiles_cols = st.columns(len(TMTypes.Tile.values()))
        for i, tile in enumerate(TMTypes.Tile.values()):
            with tiles_cols[i]:
                st.write(f"**{tile.name()}**")
                st.write(f"{self.gs.getPlayerTilesPlaced()[self.player].get(tile).getValue()}")
        
        # Display milestones
        st.subheader("Milestones")
        for milestone in self.gs.getMilestones():
            progress = milestone.checkProgress(self.gs, self.player)
            claimed = "✔" if milestone.isClaimed() else ""
            st.write(f"{milestone.getComponentName()}: {progress}/{milestone.min} {claimed}")
        
        # Display awards
        st.subheader("Awards")
        for award in self.gs.getAwards():
            progress = award.checkProgress(self.gs, self.player)
            claimed = "✔" if award.isClaimed() else ""
            st.write(f"{award.getComponentName()}: {progress} {claimed}")
    
    def update(self, gs: TMGameState):
        self.gs = gs