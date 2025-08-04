"""
Abstract base class for all features
"""

from abc import ABC, abstractmethod
from pandas.core.api import DataFrame as DataFrame

class BaseFeatures(ABC):

    @abstractmethod
    def load_data(self, season: int) -> DataFrame:
        pass
