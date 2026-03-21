"""campaign_profiles.py

Defines named comparison campaign profiles for cross-loto architecture comparison.

A campaign profile bundles all hyperparameters needed for a reproducible, named
comparison run:
  - preset (training epochs / folds / batch_size / patience — see train_prob_model.PRESET_CONFIGS)
  - seeds (list of random seeds)
  - loto_types (which lottery games to compare across)
  - evaluation_model_variants (which model variants to evaluate)
  - saved_calibration_method (calibration to save in production artifact)
  - evaluation_calibration_methods (calibration methods to compare)
  - description (human-readable summary)

Available profiles
------------------
  archcomp_lite  : Fast 2-seed, single loto_type (loto6) validation.
                   Use for quick sanity checks; NOT for final decisions.
  archcomp       : Standard 3-seed, all loto_types comparison.
                   Default for campaign decisions; balances speed and coverage.
  archcomp_full  : Extended 5-seed, all loto_types, full-epoch preset.
                   Use when signals are borderline or seeds < 5 is flagged in
                   next_experiment_recommendations.

Existing presets (smoke / fast / default / archcomp) in train_prob_model.py are
NOT changed.  Campaign profiles are a higher-level abstraction that reference
those presets.
"""

from __future__ import annotations

CAMPAIGN_PROFILE_SCHEMA_VERSION = 1

CAMPAIGN_PROFILES: dict[str, dict] = {
    "archcomp_lite": {
        "preset": "fast",
        "seeds": [42, 123],
        "loto_types": ["loto6"],
        "evaluation_model_variants": "legacy,multihot,deepsets,settransformer",
        "saved_calibration_method": "none",
        "evaluation_calibration_methods": "none,temperature,isotonic",
        "description": (
            "Fast 2-seed single-loto validation.  "
            "Quick sanity check — not for promotion decisions."
        ),
    },
    "archcomp": {
        "preset": "archcomp",
        "seeds": [42, 123, 456],
        "loto_types": ["loto6", "loto7", "miniloto"],
        "evaluation_model_variants": "legacy,multihot,deepsets,settransformer",
        "saved_calibration_method": "none",
        "evaluation_calibration_methods": "none,temperature,isotonic",
        "description": (
            "Standard 3-seed cross-loto comparison.  "
            "Default profile for campaign decisions."
        ),
    },
    "archcomp_full": {
        "preset": "default",
        "seeds": [42, 123, 456, 789, 999],
        "loto_types": ["loto6", "loto7", "miniloto"],
        "evaluation_model_variants": "legacy,multihot,deepsets,settransformer",
        "saved_calibration_method": "none",
        "evaluation_calibration_methods": "none,temperature,isotonic",
        "description": (
            "Extended 5-seed full-preset comparison.  "
            "Use when signals are borderline or run_more_seeds is recommended."
        ),
    },
}

VALID_PROFILE_NAMES: list[str] = sorted(CAMPAIGN_PROFILES.keys())


def get_profile(profile_name: str) -> dict:
    """Return a copy of the named campaign profile dict.

    Raises ValueError for unknown profile names.
    """
    if profile_name not in CAMPAIGN_PROFILES:
        raise ValueError(
            f"Unknown campaign profile: {profile_name!r}. "
            f"Valid profiles: {VALID_PROFILE_NAMES}"
        )
    return dict(CAMPAIGN_PROFILES[profile_name])


def resolve_profile(profile_name: str, overrides: dict | None = None) -> dict:
    """Return a profile dict with optional key-level overrides applied.

    Overrides replace values at the top level of the profile dict.  Unknown
    keys in overrides are passed through (they will be validated downstream).
    """
    profile = get_profile(profile_name)
    if overrides:
        profile.update({k: v for k, v in overrides.items() if v is not None})
    return profile


def list_profiles() -> None:
    """Print available campaign profiles and their descriptions to stdout."""
    print("Available campaign profiles:")
    print()
    for name in VALID_PROFILE_NAMES:
        p = CAMPAIGN_PROFILES[name]
        print(f"  {name}")
        print(f"    preset:  {p['preset']}")
        print(f"    seeds:   {p['seeds']}")
        print(f"    loto_types: {p['loto_types']}")
        print(f"    {p['description']}")
        print()
