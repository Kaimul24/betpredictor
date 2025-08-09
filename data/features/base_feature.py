"""
Abstract base class for all features
"""

from abc import ABC, abstractmethod
from pandas.core.api import DataFrame as DataFrame

class BaseFeatures(ABC):
    
    def __init__(self, season: int):
        self.season = season

    @abstractmethod
    def load_data(self) -> DataFrame:
        pass
