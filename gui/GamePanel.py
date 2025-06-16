import streamlit as st
from typing import Optional
from PIL import Image
import numpy as np

class GamePanel:
    def __init__(self):
        self.background: Optional[Image.Image] = None
        self.bg_color: Optional[str] = None
        self.keep_background_ratio: bool = True
        self.alpha: float = 0.3
        
    def set_background(self, background: Image.Image):
        """Set the background image for the panel"""
        self.background = background
        
    def set_bg_color(self, color: str):
        """Set the background color for the panel"""
        self.bg_color = color
        
    def set_keep_background_ratio(self, keep_ratio: bool):
        """Set whether to maintain the background image aspect ratio"""
        self.keep_background_ratio = keep_ratio
        
    def set_alpha(self, alpha: float):
        """Set the transparency level for the background"""
        self.alpha = alpha
        
    def display(self):
        """Render the panel using Streamlit"""
        if self.background is not None:
            # Convert PIL Image to numpy array for processing
            bg_array = np.array(self.background)
            
            # Apply alpha transparency
            if bg_array.shape[2] == 4:  # Already has alpha channel
                bg_array[:, :, 3] = (bg_array[:, :, 3] * self.alpha).astype(np.uint8)
            else:  # Add alpha channel
                alpha_channel = (np.ones(bg_array.shape[:2]) * 255 * self.alpha).astype(np.uint8)
                bg_array = np.dstack((bg_array, alpha_channel))
                
            # Create a container with the background
            with st.container():
                if self.keep_background_ratio:
                    st.image(bg_array, use_column_width=True)
                else:
                    st.image(bg_array, use_column_width=True, output_format="PNG")
                    
        elif self.bg_color is not None:
            # Use Streamlit's native theming for background color
            st.markdown(
                f"""
                <style>
                .stApp {{
                    background-color: {self.bg_color};
                }}
                </style>
                """,
                unsafe_allow_html=True
            )