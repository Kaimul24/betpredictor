"""
Abstract base class for all features
"""

from abc import ABC, abstractmethod
from pandas.core.api import DataFrame as DataFrame

# TODO: ADD ABSTRACT METHODS
class BaseFeatures(ABC):

    def __init__(self, season: int, data: DataFrame):
        self.season = season
        self.data = data

    @abstractmethod
    def load_features(self) -> DataFrame:
        pass


        