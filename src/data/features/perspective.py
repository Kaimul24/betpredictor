from collections.abc import Iterable
import pandas as pd
from src.data.features.base_feature import BaseFeatures


STARTER_COLS = [
    "starter_era",
    "starter_babip",
    "starter_hard_hit",
    "starter_k_percent",
    "starter_barrel_percent",
    "starter_fip",
    "starter_siera",
    "starter_stuff",
    "starter_ev",
    "starter_hr_fb",
    "starter_wpa",
]

RELIEVER_COLS = [
    "pen_era",
    "pen_babip",
    "pen_hard_hit",
    "pen_k_percent",
    "pen_barrel_percent",
    "pen_fip",
    "pen_siera",
    "pen_stuff",
    "pen_ev",
    "pen_hr_fb",
    "pen_wpa_li",
]

STARTER_VS_BATTER_COLS = [
    ("starter_fip", "woba"),
    ("starter_k_percent", "k_percent"),
    ("starter_bb_percent", "bb_percent"),
    ("starter_barrel_percent", "barrel_percent"),
]

TEAM_METRIC_COLS = [
    "win_pct",
    "pyth_expectation",
    "run_diff",
    "one_run_win_pct",
]

def duplicate_training_perspective(
    df: pd.DataFrame,
    *,
    market_probability_col: str = "p_open_home_median_nv",
    batter_halflives=(4, 12),
    starter_halflives=(3, 8),
    reliever_halflives=(3, 8),
    team_halflives=(3, 8, 20),
) -> pd.DataFrame:
    """Return home-focal rows plus away-focal flipped copies for training."""
    original = df.copy()
    original["focal_is_home"] = True

    flipped = df.copy()
    _swap_home_away_columns(flipped)
    flipped["is_winner_home"] = 1 - flipped["is_winner_home"]

    if market_probability_col in flipped.columns:
        flipped[market_probability_col] = 1 - flipped[market_probability_col]

    flipped["focal_is_home"] = False

    duplicated = pd.concat([original, flipped], axis=0)
    return recompute_matchup_columns(
        duplicated,
        batter_halflives=batter_halflives,
        starter_halflives=starter_halflives,
        reliever_halflives=reliever_halflives,
        team_halflives=team_halflives,
    )


def recompute_matchup_columns(
    df: pd.DataFrame,
    *,
    batter_halflives,
    starter_halflives,
    reliever_halflives,
    team_halflives,
) -> pd.DataFrame:
    """Drop FeaturePipeline matchup columns and rebuild them from home/away inputs."""
    result = df.drop(
        columns=_matchup_column_names(
            batter_halflives=batter_halflives,
            starter_halflives=starter_halflives,
            reliever_halflives=reliever_halflives,
            team_halflives=team_halflives,
        ),
        errors="ignore",
    )

    starter_ewm_cols = _ewm_cols(starter_halflives)
    reliever_ewm_cols = _ewm_cols(reliever_halflives)
    batter_ewm_cols = _ewm_cols(batter_halflives)
    team_ewm_cols = _ewm_cols(team_halflives)

    result = result.assign(
        **BaseFeatures._add_matchup_cols_diff_same_base(
            df=result,
            cols=STARTER_COLS,
            ewm_cols=starter_ewm_cols,
        )
    )
    result = result.assign(
        **BaseFeatures._add_matchup_cols_diff_same_base(
            df=result,
            cols=RELIEVER_COLS,
            ewm_cols=reliever_ewm_cols,
        )
    )
    result = result.assign(
        **BaseFeatures._add_matchup_cols_diff_base(
            df=result,
            col1=[col1 for col1, _ in STARTER_VS_BATTER_COLS],
            col2=[col2 for _, col2 in STARTER_VS_BATTER_COLS],
            col1_ewm_cols=starter_ewm_cols,
            col2_ewm_cols=batter_ewm_cols,
        )
    )
    return result.assign(
        **BaseFeatures._add_matchup_cols_diff_same_base(
            df=result,
            cols=TEAM_METRIC_COLS,
            ewm_cols=team_ewm_cols,
        )
    )


def _swap_home_away_columns(df: pd.DataFrame) -> None:
    paired_suffixes = [
        col.removeprefix("home_")
        for col in df.columns
        if col.startswith("home_") and f"away_{col.removeprefix('home_')}" in df.columns
    ]

    for suffix in paired_suffixes:
        home_col = f"home_{suffix}"
        away_col = f"away_{suffix}"
        home_values = df[home_col].copy()
        df[home_col] = df[away_col]
        df[away_col] = home_values


def _matchup_column_names(
    *,
    batter_halflives: Iterable[int],
    starter_halflives: Iterable[int],
    reliever_halflives: Iterable[int],
    team_halflives: Iterable[int],
) -> list[str]:
    starter_ewm_cols = _ewm_cols(starter_halflives)
    reliever_ewm_cols = _ewm_cols(reliever_halflives)
    batter_ewm_cols = _ewm_cols(batter_halflives)
    team_ewm_cols = _ewm_cols(team_halflives)

    names = [
        f"home_away_{col}_{ewm_col}_diff"
        for col in STARTER_COLS
        for ewm_col in starter_ewm_cols
    ]
    names.extend(
        f"home_away_{col}_{ewm_col}_diff"
        for col in RELIEVER_COLS
        for ewm_col in reliever_ewm_cols
    )
    names.extend(
        f"home_away_{col1}_{col2}_{starter_ewm_col}_diff"
        for col1, col2 in STARTER_VS_BATTER_COLS
        for starter_ewm_col, _ in zip(starter_ewm_cols, batter_ewm_cols)
    )
    names.extend(
        f"home_away_{col}_{ewm_col}_diff"
        for col in TEAM_METRIC_COLS
        for ewm_col in team_ewm_cols
    )
    return names


def _ewm_cols(halflives: Iterable[int]) -> list[str]:
    return ["season", *[f"ewm_h{halflife}" for halflife in halflives]]
