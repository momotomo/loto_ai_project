import argparse
import csv
import json
from datetime import date, datetime, timedelta
from io import StringIO
from urllib.request import urlopen
from zoneinfo import ZoneInfo


HOLIDAY_CSV_URL = "https://www8.cao.go.jp/chosei/shukujitsu/syukujitsu.csv"
JST = ZoneInfo("Asia/Tokyo")
DRAW_WEEKDAYS = {
    "miniloto": {1},
    "loto6": {0, 3},
    "loto7": {4},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Compute lottery targets that should kick on the next business day after draw day.")
    parser.add_argument("--today", help="JST date override in YYYY-MM-DD format")
    parser.add_argument("--output", help="Optional path to write the JSON result")
    parser.add_argument("--holiday-csv-url", default=HOLIDAY_CSV_URL)
    return parser.parse_args()


def resolve_today(today_text):
    if today_text:
        return datetime.strptime(today_text, "%Y-%m-%d").date()
    return datetime.now(JST).date()


def decode_holiday_csv(raw_bytes):
    for encoding in ("cp932", "shift_jis", "utf-8"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Failed to decode the Cabinet Office holiday CSV.")


def load_holiday_set(csv_url):
    with urlopen(csv_url, timeout=30) as response:
        csv_text = decode_holiday_csv(response.read())

    reader = csv.reader(StringIO(csv_text))
    next(reader, None)
    holidays = set()
    for row in reader:
        if not row:
            continue
        holidays.add(datetime.strptime(row[0].strip(), "%Y/%m/%d").date())
    return holidays


def is_business_day(target_date, holidays):
    return target_date.weekday() < 5 and target_date not in holidays


def find_last_draw_day(today, weekday_set):
    candidate = today - timedelta(days=1)
    while candidate.weekday() not in weekday_set:
        candidate -= timedelta(days=1)
    return candidate


def next_business_day(start_date, holidays):
    candidate = start_date + timedelta(days=1)
    while not is_business_day(candidate, holidays):
        candidate += timedelta(days=1)
    return candidate


def compute_targets(today, holidays, holiday_source):
    targets = []
    details = {}

    for loto_type, weekday_set in DRAW_WEEKDAYS.items():
        last_draw = find_last_draw_day(today, weekday_set)
        kick_day = next_business_day(last_draw, holidays)
        should_kick = kick_day == today
        if should_kick:
            targets.append(loto_type)
        details[loto_type] = {
            "last_draw": last_draw.isoformat(),
            "kick_day": kick_day.isoformat(),
            "should_kick": should_kick,
        }

    return {
        "today": today.isoformat(),
        "targets": targets,
        "details": details,
        "holiday_source": holiday_source,
    }


def main():
    args = parse_args()
    today = resolve_today(args.today)
    holidays = load_holiday_set(args.holiday_csv_url)
    result = compute_targets(today, holidays, args.holiday_csv_url)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")


if __name__ == "__main__":
    main()
