import pandas as pd
import pytest

from data_collector import DataCollectionError, validate_history_dataframe


def build_valid_loto6_history():
    return pd.DataFrame(
        [
            {"draw_id": 1, "date": "2024/01/04", "num1": 1, "num2": 2, "num3": 3, "num4": 4, "num5": 5, "num6": 6},
            {"draw_id": 2, "date": "2024/01/11", "num1": 7, "num2": 8, "num3": 9, "num4": 10, "num5": 11, "num6": 12},
            {"draw_id": 2, "date": "2024/01/11", "num1": 13, "num2": 14, "num3": 15, "num4": 16, "num5": 17, "num6": 18},
            {"draw_id": 3, "date": "2024/01/18", "num1": 19, "num2": 20, "num3": 21, "num4": 22, "num5": 23, "num6": 24},
        ]
    )


def test_duplicate_draw_id_is_detected_and_removed():
    validated, removed_duplicates = validate_history_dataframe(build_valid_loto6_history(), "loto6")

    assert removed_duplicates == 1
    assert validated["draw_id"].tolist() == [1, 2, 3]
    assert validated.iloc[1]["num1"] == 13


@pytest.mark.parametrize(
    ("column_name", "invalid_value", "message_fragment"),
    [
        ("draw_id", "x", "draw_id を数値として解釈できない"),
        ("num3", "oops", "num3 を数値として解釈できない"),
    ],
)
def test_invalid_numeric_format_raises(column_name, invalid_value, message_fragment):
    invalid_df = build_valid_loto6_history()
    invalid_df[column_name] = invalid_df[column_name].astype(object)
    invalid_df.loc[0, column_name] = invalid_value

    with pytest.raises(DataCollectionError, match=message_fragment):
        validate_history_dataframe(invalid_df, "loto6")


def test_out_of_range_number_raises():
    invalid_df = build_valid_loto6_history()
    invalid_df.loc[0, "num6"] = 99

    with pytest.raises(DataCollectionError, match="数字範囲外"):
        validate_history_dataframe(invalid_df, "loto6")
