import streamlit as st
import math
from games.terraformingmars import TMGameState, TMTypes
from games.terraformingmars.components import TMCard
from games.terraformingmars.actions import (
    TMAction, ModifyPlayerResource, PlaceTile, CompoundAction
)
from games.terraformingmars.rules.effects import Effect, PayForActionEffect
from games.terraformingmars.rules.requirements import (
    TagsPlayedRequirement, TagOnCardRequirement
)
from games.terraformingmars.rules import Discount

class TMCardView:
    def __init__(self, gs: TMGameState, card: TMCard, index: int, width: int, height: int):
        self.gs = gs
        self.card = card
        self.index = index
        self.width = width
        self.height = height
        self.spacing = 10
        self.clicked = False
        self.views = None
        self.gui = None
        
        # Colors
        self.any_player_color = (234, 38, 38, 168)
        self.default_item_size = 30
        
        # Placeholder for images - would need actual game assets
        self.point_bg = None  
        self.proj_card_bg = None
        self.production_img = None
        self.action_arrow = None
        self.req_min = None
        self.req_max = None
        
        self.above_ribbon = (width//5, 0, width - width//5 - self.spacing//2, height//8)
        
    def inform_gui(self, gui):
        self.gui = gui
        
    def inform_other_views(self, views):
        self.views = views
        
    def display(self):
        if self.card is None:
            return
            
        if self.card.card_type == TMTypes.CardType.Corporation:
            self._draw_corporation_card()
        else:
            self._draw_project_card()
            
    def _draw_project_card(self):
        # Create card container
        with st.container():
            # Card background
            st.markdown(f"<div style='background-color:#f0f0f0; border-radius:10px; padding:10px; width:{self.width}px; height:{self.height}px;'>", 
                       unsafe_allow_html=True)
            
            # Card ribbon with type
            st.markdown(f"<div style='background-color:{self._get_card_type_color()}; padding:5px; border-radius:5px;'>", 
                       unsafe_allow_html=True)
            st.markdown(f"<h4 style='text-align:center;'>{self.card.get_component_name()}</h4>", 
                       unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Cost in top-left
            st.markdown(f"<div style='position:absolute; top:10px; left:10px;'>Cost: {self.card.cost}</div>", 
                       unsafe_allow_html=True)
            
            # Points in bottom-right
            if self.card.n_points != 0:
                self._draw_points()
                
            # Tags above ribbon
            self._draw_tags()
            
            # Requirements
            if self.card.requirements:
                self._draw_requirements()
                
            # Resources on card
            if self.card.resource_on_card:
                self._draw_resources_on_card()
                
            # Actions
            for action in self.card.actions:
                self._draw_action(action)
                
            # Discounts
            for discount in self.card.discount_effects:
                self._draw_discount(discount)
                
            # Resource mappings
            for rm in self.card.resource_mappings:
                self._draw_resource_mapping(rm)
                
            # After-action effects
            for effect in self.card.persisting_effects:
                self._draw_effect(effect)
                
            # Immediate effects
            for effect in self.card.immediate_effects:
                self._draw_card_effect(effect)
                
            st.markdown("</div>", unsafe_allow_html=True)
            
    def _draw_corporation_card(self):
        # Similar structure to project card but with corporation-specific layout
        with st.container():
            st.markdown(f"<div style='background-color:#e0e0ff; border-radius:10px; padding:10px; width:{self.width}px; height:{self.height}px;'>", 
                       unsafe_allow_html=True)
            
            # Corporation name
            st.markdown(f"<h4 style='text-align:center;'>{self.card.get_component_name()}</h4>", 
                       unsafe_allow_html=True)
            
            # Tags
            self._draw_tags()
            
            # Starting resources
            for effect in self.card.immediate_effects:
                if isinstance(effect, ModifyPlayerResource):
                    self._draw_modify_player_resource(effect)
            
            # First action
            if self.card.first_action:
                self._draw_action(self.card.first_action, is_first=True)
                
            # Other actions
            for action in self.card.actions:
                self._draw_action(action)
                
            # Discounts, resource mappings, effects would go here...
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    def _draw_points(self):
        if self.card.points_resource:
            cols = st.columns([1,2])
            with cols[0]:
                st.write(f"1/{int(1/self.card.n_points)}")
            with cols[1]:
                st.image(self._get_resource_image(self.card.points_resource), width=30)
        elif self.card.points_tile:
            cols = st.columns([1,2])
            with cols[0]:
                st.write(f"1/{int(1/self.card.n_points)}")
            with cols[1]:
                st.image(self._get_tile_image(self.card.points_tile), width=30)
        elif self.card.points_tag:
            cols = st.columns([1,2])
            with cols[0]:
                st.write(f"1/{int(1/self.card.n_points)}")
            with cols[1]:
                st.image(self._get_tag_image(self.card.points_tag), width=30)
        else:
            st.write(f"Points: {self.card.n_points}")
    
    def _draw_tags(self):
        if self.card.tags:
            cols = st.columns(len(self.card.tags))
            for i, tag in enumerate(self.card.tags):
                with cols[i]:
                    st.image(self._get_tag_image(tag), width=30)
    
    def _draw_requirements(self):
        has_max = any(req.is_max() for req in self.card.requirements)
        
        with st.expander("Requirements", expanded=True):
            for req in self.card.requirements:
                text = req.get_display_text(self.gs)
                images = req.get_display_images()
                
                if text:
                    st.write(text)
                if images:
                    cols = st.columns(len(images))
                    for i, img in enumerate(images):
                        with cols[i]:
                            st.image(img, width=30)
    
    def _draw_resources_on_card(self):
        if self.card.resource_on_card:
            cols = st.columns(3)
            with cols[0]:
                st.image(self._get_resource_image(self.card.resource_on_card), width=30)
            with cols[1]:
                st.write(":")
            with cols[2]:
                st.write(str(self.card.n_resources_on_card))
    
    def _draw_action(self, action: TMAction, is_first: bool = False):
        if is_first:
            st.write("*First Action*")
            
        if isinstance(action, ModifyPlayerResource):
            self._draw_modify_player_resource(action)
        elif isinstance(action, PlaceTile):
            self._draw_place_tile(action)
        # Handle other action types...
    
    def _draw_modify_player_resource(self, action: ModifyPlayerResource):
        change = int(action.change) if action.change == math.floor(action.change) else action.change
        cols = st.columns(3)
        with cols[0]:
            st.write(str(change))
        with cols[1]:
            st.write("→")
        with cols[2]:
            st.image(self._get_resource_image(action.resource), width=30)
            
        if action.production:
            st.write("(production)")
        if action.opponents:
            st.write("*affects opponents*")
    
    def _draw_place_tile(self, action: PlaceTile):
        cols = st.columns(3)
        with cols[0]:
            st.image(self._get_tile_image(action.tile), width=30)
        with cols[1]:
            st.write("→")
        with cols[2]:
            if action.map_type:
                st.write(action.map_type.name())
            if action.tile_name:
                st.write(action.tile_name)
    
    def _draw_discount(self, discount: Discount):
        req = discount.requirement
        amount = discount.amount
        
        if isinstance(req, (TagsPlayedRequirement, TagOnCardRequirement)):
            tags = req.tags if hasattr(req, 'tags') else []
            cols = st.columns(len(tags) + 3)
            for i, tag in enumerate(tags):
                with cols[i]:
                    st.image(self._get_tag_image(tag), width=30)
            with cols[-3]:
                st.write(":")
            with cols[-2]:
                st.write(f"-{amount}")
            with cols[-1]:
                st.image(self._get_resource_image(TMTypes.Resource.MegaCredit), width=30)
        # Handle other requirement types...
    
    def _draw_resource_mapping(self, rm):
        from_img = self._get_resource_image(rm.from_res)
        to_img = self._get_resource_image(rm.to_res)
        
        cols = st.columns(5)
        with cols[0]:
            st.image(from_img, width=30)
        with cols[1]:
            st.write(":")
        with cols[2]:
            rate = int(rm.rate) if rm.rate == math.floor(rm.rate) else rm.rate
            st.write(str(rate))
        with cols[3]:
            st.write("→")
        with cols[4]:
            st.image(to_img, width=30)
    
    def _draw_effect(self, effect: Effect):
        if isinstance(effect, PayForActionEffect):
            cols = st.columns(3)
            with cols[0]:
                st.write(str(effect.min_cost))
            with cols[1]:
                st.image(self._get_resource_image(TMTypes.Resource.MegaCredit), width=30)
            with cols[2]:
                st.write(":")
            # Draw effect action...
        # Handle other effect types...
    
    def _draw_card_effect(self, action: TMAction):
        if isinstance(action, ModifyPlayerResource):
            self._draw_modify_player_resource(action)
        elif isinstance(action, CompoundAction):
            for sub_action in action.actions:
                self._draw_card_effect(sub_action)
        # Handle other action types...
    
    def _get_card_type_color(self):
        # Return color based on card type
        return {
            TMTypes.CardType.Corporation: "#aaccff",
            TMTypes.CardType.Active: "#ffcccc",
            TMTypes.CardType.Automated: "#ccffcc",
            TMTypes.CardType.Event: "#ccccff"
        }.get(self.card.card_type, "#ffffff")
    
    def _get_resource_image(self, resource: TMTypes.Resource):
        # Placeholder - would return actual image path
        return None
    
    def _get_tag_image(self, tag: TMTypes.Tag):
        # Placeholder - would return actual image path
        return None
    
    def _get_tile_image(self, tile: TMTypes.Tile):
        # Placeholder - would return actual image path
        return None
    
    def update(self, gs: TMGameState, card: TMCard, index: int):
        self.gs = gs
        self.card = card
        self.index = index
        self.clicked = False