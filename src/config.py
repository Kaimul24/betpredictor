from dataclasses import asdict, dataclass, fields, replace
from datetime import date
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent

DATABASE_PATH = PROJECT_ROOT / 'src' / 'data' / 'mlb_stats.sqlite'
SCHEMA_PATH = PROJECT_ROOT / 'src' / 'scrapers' / 'schema.sql'
FEATURES_CACHE_PATH = PROJECT_ROOT / 'src' / 'data' / 'features' 


def _namespace_values(namespace: Any) -> dict[str, Any]:
    return vars(namespace).copy()

def _select_dataclass_values(cls, values: dict[str, Any]) -> dict[str, Any]:
    field_names = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in field_names}

class ConfigMixin:
    def to_dict(self) -> dict[str, Any]:
        def convert(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, tuple):
                return [convert(item) for item in value]
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        return convert(asdict(self))


@dataclass(frozen=True)
class FeatureConfig(ConfigMixin):
    stage: str
    training_mode: str = "market_residual"
    model_type: str = "mlp"
    perspective_duplication: bool = False
    force_recreate: bool = False
    force_recreate_preprocessing: bool = False
    clear_log: bool = False
    log: bool = False
    log_file: str | None = None
    log_level: str = "info"
    file_log_level: str = "debug"
    batter_halflives: tuple[int, ...] = (4, 12)
    starter_halflives: tuple[int, ...] = (3, 8)
    reliever_halflives: tuple[int, ...] = (3, 8)
    team_halflives: tuple[int, ...] = (3, 8, 20)

    def __post_init__(self):
        if self.training_mode not in {"market_residual", "baseball_only"}:
            raise ValueError("FeatureConfig training_mode must be market_residual or baseball_only")
        if self.stage not in {"finetune", "pretrain"}:
            raise ValueError("FeatureConfig stage must be finetune or pretrain")
        if self.model_type not in {"xgboost", "mlp"}:
            raise ValueError("FeatureConfig model_type must be xgboost or mlp")
        for field_name in (
            "batter_halflives",
            "starter_halflives",
            "reliever_halflives",
            "team_halflives",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))

    @classmethod
    def from_namespace(cls, namespace: Any) -> "FeatureConfig":
        values = _select_dataclass_values(cls, _namespace_values(namespace))
        return cls(**values)


@dataclass(frozen=True)
class XGBoostConfig(ConfigMixin):
    stage: str
    training_mode: str = "stacked"
    perspective_duplication: bool = False
    retune: bool = False
    force_recreate: bool = False
    force_recreate_preprocessing: bool = False
    clear_log: bool = False
    log: bool = False
    log_file: str | None = None
    batter_halflives: tuple[int, ...] = (4, 12)
    starter_halflives: tuple[int, ...] = (3, 8)
    reliever_halflives: tuple[int, ...] = (3, 8)
    team_halflives: tuple[int, ...] = (3, 8, 20)

    def __post_init__(self):
        if self.training_mode not in {"stacked", "market_residual", "baseball_only"}:
            raise ValueError("XGBoostConfig training_mode must be stacked, market_residual, or baseball_only")
        if self.stage not in {"finetune", "pretrain"}:
            raise ValueError("XGBoostConfig stage must be finetune or pretrain")
        for field_name in (
            "batter_halflives",
            "starter_halflives",
            "reliever_halflives",
            "team_halflives",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))

    @classmethod
    def from_namespace(cls, namespace: Any) -> "XGBoostConfig":
        values = _select_dataclass_values(cls, _namespace_values(namespace))
        return cls(**values)

    def for_stage(self, training_mode: str, stage: str) -> "XGBoostConfig":
        return replace(self, training_mode=training_mode, stage=stage)

    def to_feature_config(
        self,
        stage: str,
        training_mode: str,
        model_type: str = "xgboost",
    ) -> FeatureConfig:
        return FeatureConfig(
            training_mode=training_mode,
            stage=stage,
            model_type=model_type,
            perspective_duplication=self.perspective_duplication,
            force_recreate=self.force_recreate,
            force_recreate_preprocessing=self.force_recreate_preprocessing,
            clear_log=self.clear_log,
            log=self.log,
            log_file=self.log_file,
            batter_halflives=self.batter_halflives,
            starter_halflives=self.starter_halflives,
            reliever_halflives=self.reliever_halflives,
            team_halflives=self.team_halflives,
        )


@dataclass(frozen=True)
class NeuralNetworkConfig(ConfigMixin):
    stage: str
    training_mode: str = "market_residual"
    perspective_duplication: bool = False
    base_hidden_size: int = 256
    max_residual: float = 0.5
    alpha: float = 0.7
    p_drop: float = 0.2
    train_batch: int = 1024
    val_batch: int = 8192
    epochs: int = 100
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.03
    cosine_scheduler: bool = True
    retune: bool = False
    use_hyperparams: bool = False
    force_recreate: bool = False
    force_recreate_preprocessing: bool = False
    clear_log: bool = False
    log: bool = False
    log_file: str | None = None
    batter_halflives: tuple[int, ...] = (4, 12)
    starter_halflives: tuple[int, ...] = (3, 8)
    reliever_halflives: tuple[int, ...] = (3, 8)
    team_halflives: tuple[int, ...] = (3, 8, 20)

    def __post_init__(self):
        if self.training_mode not in {"stacked", "market_residual", "baseball_only"}:
            raise ValueError("NeuralNetworkConfig training_mode must be stacked, market_residual, or baseball_only")
        if self.stage not in {"finetune", "pretrain"}:
            raise ValueError("NeuralNetworkConfig stage must be finetune or pretrain")
        for field_name in (
            "batter_halflives",
            "starter_halflives",
            "reliever_halflives",
            "team_halflives",
        ):
            object.__setattr__(self, field_name, tuple(getattr(self, field_name)))

    @classmethod
    def from_namespace(cls, namespace: Any) -> "NeuralNetworkConfig":
        values = _select_dataclass_values(cls, _namespace_values(namespace))
        return cls(**values)

    def for_stage(self, training_mode: str, stage: str, perspective_duplication: bool) -> "NeuralNetworkConfig":
        return replace(self, training_mode=training_mode, stage=stage, perspective_duplication=perspective_duplication)
    
    def to_feature_config(
        self,
        stage: str,
        training_mode: str,
        model_type: str = "mlp",
    ) -> FeatureConfig:
        return FeatureConfig(
            training_mode=training_mode,
            stage=stage,
            model_type=model_type,
            perspective_duplication=self.perspective_duplication,
            force_recreate=self.force_recreate,
            force_recreate_preprocessing=self.force_recreate_preprocessing,
            clear_log=self.clear_log,
            log=self.log,
            log_file=self.log_file,
            batter_halflives=self.batter_halflives,
            starter_halflives=self.starter_halflives,
            reliever_halflives=self.reliever_halflives,
            team_halflives=self.team_halflives,
        )


@dataclass(frozen=True)
class TwoHeadNNConfig(NeuralNetworkConfig):
    device: str = "auto"
    pretrain_trials: int = 50
    pretrain_dir: Path = PROJECT_ROOT / "src" / "data" / "models" / "saved_models" / "nn_pretrain_ckpts"
    finetune_dir: Path = PROJECT_ROOT / "src" / "data" / "models" / "saved_models" / "nn_finetune_ckpts"
    pretrained_checkpoint: Path | None = None
    encoder_freeze_epochs: int = 10
    residual_penalty: float = 0.05

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, "pretrain_dir", Path(self.pretrain_dir))
        object.__setattr__(self, "finetune_dir", Path(self.finetune_dir))
        object.__setattr__(self, "encoder_freeze_epochs", self.encoder_freeze_epochs)
        object.__setattr__(self, "residual_penalty", self.residual_penalty)
        if self.pretrained_checkpoint is not None:
            object.__setattr__(self, "pretrained_checkpoint", Path(self.pretrained_checkpoint))
    @classmethod
    def from_namespace(cls, namespace: Any) -> "TwoHeadNNConfig":
        values = _select_dataclass_values(cls, _namespace_values(namespace))
        return cls(**values)

TEAM_TO_TEAM_ID_STATSAPI_MAP = {
    'LAA': 108,
    'ARI': 109,
    'BAL': 110,
    'BOS': 111,
    'CHC': 112,
    'CIN': 113,
    'CLE': 114,
    'COL': 115,
    'DET': 116,
    'HOU': 117,
    'KCR': 118,
    'LAD': 119,
    'WSN': 120,
    'NYM': 121,
    'OAK': 133,
    'ATH': 133,
    'PIT': 134,
    'SDP': 135,
    'SEA': 136,
    'SFG': 137,
    'STL': 138,
    'TBR': 139,
    'TEX': 140,
    'TOR': 141,
    'MIN': 142,
    'PHI': 143,
    'ATL': 144,
    'CHW': 145,
    'MIA': 146,
    'NYY': 147,
    'MIL': 158
}

INT_TO_MONTH_MAP_FIELDING = {
    4: 'Mar/Apr',
    5: 'May',
    6: 'Jun',
    7: 'Jul',
    8: 'Aug',
    9: 'Sep/Oct',
}

DATES = {
    '2016': [date(2016, 4, 3), date(2016, 10, 2)],
    '2017': [date(2017, 4, 2), date(2017, 10, 1)],
    '2018': [date(2018, 3, 29), date(2018, 10, 1)],
    '2019': [date(2019, 3, 20), date(2019, 9, 29)],
    '2021': [date(2021, 4, 1), date(2021, 10, 4)],
    '2022': [date(2022, 4, 7), date(2022, 10, 6)],
    '2023': [date(2023, 3, 30), date(2023, 10, 2)],
    '2024': [date(2024, 3, 28), date(2024, 10, 1)],
    '2025': [date(2025, 3, 18), date(2025, 9, 29)]
}

LG_AVG_STATS = {
    "2021": {"Bats": {"ops": 0.7278624, "babip": 0.291664854, "bb_k": 0.374753826, "woba": 0.31442644240601775, "barrel_percent": 0.07911897, "hard_hit": 0.38520605, "ev": 88.76291668581086, "iso": 0.166955867, "gb_fb": 1.176134806}, "Throws": {"era": 4.265892667856131, "babip": 0.2895822900657906, "k_percent": 0.23179901, "bb_percent": 0.08686756, "barrel_percent": 0.07911897, "hard_hit": 0.38520605, "siera": 4.175190307395692, "fip": 4.265892485031086, "ev": 88.76291679304866, "hr_fb": 0.135720157, "gmli": 1.0584131599484485}},
    "2022": {"Bats": {"ops": 0.706412709, "babip": 0.290404678, "bb_k": 0.363937077, "woba": 0.30974585067181093, "barrel_percent": 0.07502755, "hard_hit": 0.38169228, "ev": 88.57476170256407, "iso": 0.152148778, "gb_fb": 1.153070137}, "Throws": {"era": 3.96832714563721, "babip": 0.28927354229974983, "k_percent": 0.22417771, "bb_percent": 0.08158658, "barrel_percent": 0.07502755, "hard_hit": 0.38169228, "siera": 3.8756488351581293, "fip": 3.9683269469843023, "ev": 88.57476181142745, "hr_fb": 0.113874574, "gmli": 1.057056432586794}},
    "2023": {"Bats": {"ops": 0.734292472, "babip": 0.296522719, "bb_k": 0.378056066, "woba": 0.31837469936833196, "barrel_percent": 0.08060002, "hard_hit": 0.39204536, "ev": 89.00106155112319, "iso": 0.165772604, "gb_fb": 1.134658315}, "Throws": {"era": 4.331714250563033, "babip": 0.2952085900964022, "k_percent": 0.22727915, "bb_percent": 0.08592426, "barrel_percent": 0.08060002, "hard_hit": 0.39204536, "siera": 4.237841443447046, "fip": 4.331714183169491, "ev": 89.00106150666907, "hr_fb": 0.127180909, "gmli": 1.0809939454776771}},
    "2024": {"Bats": {"ops": 0.711329941, "babip": 0.290537456, "bb_k": 0.362380755, "woba": 0.310181047531053, "barrel_percent": 0.07797881, "hard_hit": 0.38651521, "ev": 88.83302249496766, "iso": 0.155931748, "gb_fb": 1.108267212}, "Throws": {"era": 4.07894181840083,  "babip": 0.28918805216659654, "k_percent": 0.22580009, "bb_percent": 0.08182561, "barrel_percent": 0.07797881, "hard_hit": 0.38651521, "siera": 3.9892671273186298, "fip": 4.078941491380367, "ev": 88.8330223533932, "hr_fb": 0.116308335, "gmli": 1.0586196697086832}},
    "2025": {"Bats": {"ops": 0.720503116, "babip": 0.29076157, "bb_k": 0.384028185, "woba": 0.3140384167056315, "barrel_percent": 0.08537562, "hard_hit": 0.40674261, "ev": 89.36394618911095, "iso": 0.15841141, "gb_fb": 1.082638906}, "Throws": {"era": 4.1649425807328955, "babip": 0.2891725525786213, "k_percent": 0.21985541, "bb_percent": 0.08443067, "barrel_percent": 0.08537562, "hard_hit": 0.40674261, "siera": 4.065775345315857, "fip": 4.164942388606765, "ev": 89.36394622448286, "hr_fb": 0.117677604, "gmli": 1.0683916095408696}}
}

POSITION_MAP = {
    '1': 'P',
    '2': 'C',
    '3': '1B',
    '4': '2B',
    '5': '3B',
    '6': 'SS',
    '7': 'LF',
    '8': 'CF',
    '9': 'RF'
}

TEAM_ABBR_MAP = {
    'Arizona Diamondbacks': 'ARI',
    'Atlanta Braves': 'ATL',
    'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS',
    'Chicago Cubs': 'CHC',
    'Chicago White Sox': 'CHW',
    'Cincinnati Reds': 'CIN',
    'Cleveland Guardians': 'CLE',
    'Cleveland Indians': 'CLE',
    'Colorado Rockies': 'COL',
    'Detroit Tigers': 'DET',
    'Houston Astros': 'HOU',
    'Kansas City Royals': 'KCR',
    'Los Angeles Angels': 'LAA',
    'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA',
    'Milwaukee Brewers': 'MIL',
    'Minnesota Twins': 'MIN',
    'New York Mets': 'NYM',
    'New York Yankees': 'NYY',
    'Oakland Athletics': 'ATH',
    'Athletics': 'ATH', # MAY NEED TO CHANGE
    'OAK': 'ATH',
    'Philadelphia Phillies': 'PHI',
    'Pittsburgh Pirates': 'PIT',
    'San Diego Padres': 'SDP',
    'San Francisco Giants': 'SFG',
    'Seattle Mariners': 'SEA',
    'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TBR',
    'Texas Rangers': 'TEX',
    'Toronto Blue Jays': 'TOR',
    'Washington Nationals': 'WSN',
    'Diamondbacks': 'ARI',
    'Braves': 'ATL',
    'Orioles': 'BAL',
    'Red Sox': 'BOS',
    'Cubs': 'CHC',
    'White Sox': 'CHW',
    'Reds': 'CIN',
    'Cleveland': 'CLE',
    'Guardians': 'CLE',
    'Indians': 'CLE',
    'Rockies': 'COL',
    'Tigers': 'DET',
    'Astros': 'HOU',
    'Royals': 'KCR',
    'Angels': 'LAA',
    'Dodgers': 'LAD',
    'Marlins': 'MIA',
    'Brewers': 'MIL',
    'Twins': 'MIN',
    'Mets': 'NYM',
    'Yankees': 'NYY',
    'Athletics': 'ATH',
    'Athletics': 'ATH',
    'Phillies': 'PHI',
    'Pirates': 'PIT',
    'Padres': 'SDP',
    'Giants': 'SFG',
    'Mariners': 'SEA',
    'Cardinals': 'STL',
    'Rays': 'TBR',
    'Rangers': 'TEX',
    'Blue Jays': 'TOR',
    'Nationals': 'WSN',
}

ODDS_TEAM_ABBR_MAP = {
    'AZ': 'ARI',
    'KC': 'KCR',
    'OAK': 'ATH',
    'SD': 'SDP',
    'SF': 'SFG',
    'TB': 'TBR',
    'WAS': 'WSN'
}
