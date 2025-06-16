from abc import ABC, abstractmethod

class IScreenHighlight(ABC):
    @abstractmethod
    def clear_highlights(self):
        """Clear any highlights or selections on the screen"""
        pass