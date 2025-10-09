# mse_data_extractor.py

import logging
import os
import re
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import pdfplumber

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================================
# GLOBAL VARIABLES
# ===============================================
_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

# Counter lists for different periods
COUNTER_LIST = {
    "2017": [
        "BHL",
        "FMBTS CA CD",
        "ILLOVO",
        "MPICO TS",
        "NBM",
        "NBS",
        "NICO TS",
        "NITL TS",
        "PCL XD",
        "STANDARD",
        "Sunbird TS",
        "TNM TS CD",
        "OML",
    ],
    "2018": [
        "BHL",
        "FMBTS",
        "ILLOVO",
        "MPICO",
        "NBM",
        "NBS",
        "NICO",
        "NITL",
        "PCL",
        "STANDARD",
        "SUNBIRD",
        "TNM",
        "OML",
    ],
    "2019": [
        "BHL",
        "FMBCH",
        "ICON",
        "ILLOVO",
        "MPICO",
        "NBM",
        "NBS",
        "NICO",
        "NITL",
        "OMU",
        "PCL",
        "STANDARD",
        "SUNBIRD",
        "TNM",
    ],
    "2020": [
        "AIRTEL",
        "BHL",
        "FMBCH",
        "ICON",
        "ILLOVO",
        "MPICO",
        "NBM",
        "NBS",
        "NICO",
        "NITL",
        "OMU",
        "PCL",
        "STANDARD",
        "SUNBIRD",
        "TNM",
    ],
    "2021-2025": [
        "AIRTEL",
        "BHL",
        "FDHB",
        "FMBCH",
        "ICON",
        "ILLOVO",
        "MPICO",
        "NBM",
        "NBS",
        "NICO",
        "NITL",
        "OMU",
        "PCL",
        "STANDARD",
        "SUNBIRD",
        "TNM",
    ],
}
# Column lists for different periods
COLS = {
    "2017": [
        "counter_id",
        "daily_range_high",
        "daily_range_low",
        "counter",
        "buy_price",
        "sell_price",
        "previous_closing_price",
        "today_closing_price",
        "volume_traded",
        "dividend_mk",
        "dividend_yield_pct",
        "earnings_yield_pct",
        "pe_ratio",
        "pbv_ratio",
        "market_capitalization_mkmn",
        "profit_after_tax_mkmn",
        "num_shares_issue",
    ],
    "2018": [
        "counter_id",
        "daily_range_high",
        "daily_range_low",
        "counter",
        "buy_price",
        "sell_price",
        "previous_closing_price",
        "today_closing_price",
        "volume_traded",
        "dividend_mk",
        "dividend_yield_pct",
        "earnings_yield_pct",
        "pe_ratio",
        "pbv_ratio",
        "market_capitalization_mkmn",
        "profit_after_tax_mkmn",
        "num_shares_issue",
    ],
    "2019": [
        "counter_id",
        "daily_range_high",
        "daily_range_low",
        "counter",
        "buy_price",
        "sell_price",
        "previous_closing_price",
        "today_closing_price",
        "volume_traded",
        "dividend_yield_pct",
        "earnings_yield_pct",
        "pe_ratio",
        "market_capitalization_mkmn",
        "num_shares_issue",
        "profit_after_tax_mkmn",
        "pbv_ratio",
    ],
    "2020": [
        "counter_id",
        "daily_range_high",
        "daily_range_low",
        "counter",
        "buy_price",
        "sell_price",
        "previous_closing_price",
        "today_closing_price",
        "volume_traded",
        "dividend_yield_pct",
        "earnings_yield_pct",
        "pe_ratio",
        "market_capitalization_mkmn",
        "num_shares_issue",
        "profit_after_tax_mkmn",
        "pbv_ratio",
    ],
    "2021-2025": [
        "counter_id",
        "daily_range_high",
        "daily_range_low",
        "counter",
        "buy_price",
        "sell_price",
        "previous_closing_price",
        "today_closing_price",
        "volume_traded",
        "dividend_mk",
        "dividend_yield_pct",
        "earnings_yield_pct",
        "pe_ratio",
        "pbv_ratio",
        "market_capitalization_mkmn",
        "profit_after_tax_mkmn",
        "num_shares_issue",
    ],
}

# Column mapping for 2017-2018 to 2021-2025 format
COLUMN_MAPPING_2017_2018 = {
    "Daily Range High (t)": "daily_range_high",
    "Daily Range Low (t)": "daily_range_low",
    "MSE Code": "counter",
    "Buy (t)": "buy_price",
    "Sell (t)": "sell_price",
    "Prev. Closing": "previous_closing_price",
    "Today’s Closing": "today_closing_price",
    "Volume": "volume_traded",
    "Dividend": "dividend_mk",
    "Net Yield (%)": "dividend_yield_pct",
    "Earnings Yield (%)": "earnings_yield_pct",
    "P/E Ratio": "pe_ratio",
    "P/BV Ratio": "pbv_ratio",
    "Market Capitalisation MKmn": "market_capitalization_mkmn",
    "After Tax Profit MKmn": "profit_after_tax_mkmn",
    "No. of Shares in issue": "num_shares_issue",
}

# Column mapping for 2019-2020 to 2021-2025 format
COLUMN_MAPPING_2019_2020 = {
    "No.": "counter_id",
    "Daily Range High": "daily_range_high",
    "Daily Range Low": "daily_range_low",
    "MSE Code": "counter",
    "BUY": "buy_price",
    "SELL": "sell_price",
    "Previous Closing Price": "previous_closing_price",
    "Today Closing Price": "today_closing_price",
    "Volume": "volume_traded",
    "Dividend Yield (%)": "dividend_yield_pct",
    "Net Earnings Yield (%)": "earnings_yield_pct",
    "P/E Ratio": "pe_ratio",
    "Market Capitalisation (MKmn)": "market_capitalization_mkmn",
    "No. of Shares in issue": "num_shares_issue",
    "After Tax Profit (MKmn)": "profit_after_tax_mkmn",
    "P/BV Ratio": "pbv_ratio",
}

def _mkdate(y, m, d):  # y,m,d may be str
    return date(int(y), int(m), int(d))


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _parse_date_str(s: str, day_first: bool = True):
    s = _norm_text(s)
    m = re.search(
        r"(?i)\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9}),?\s+(20\d{2})\b", s
    )
    if m:
        d, mon, y = m.groups()
        mon_num = _MONTHS.get(mon.lower())
        if mon_num:
            return _mkdate(y, mon_num, d)
    m = re.search(
        r"(?i)\b([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(20\d{2})\b", s
    )
    if m:
        mon, d, y = m.groups()
        mon_num = _MONTHS.get(mon.lower())
        if mon_num:
            return _mkdate(y, mon_num, d)
    m = re.search(r"\b(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b", s)
    if m:
        y, mth, d = m.groups()
        try:
            return _mkdate(y, mth, d)
        except ValueError:
            pass
    m = re.search(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](20\d{2})\b", s)
    if m:
        a, b, y = m.groups()
        d, mth = (a, b) if day_first else (b, a)
        try:
            return _mkdate(y, mth, d)
        except ValueError:
            pass
    return None


def extract_date_from_filename(filename):
    filename = Path(filename).name
    pattern1 = r"Daily_Report_(\d{1,2})_([A-Za-z]+)_(\d{4})\.pdf"
    match = re.search(pattern1, filename)
    if match:
        day, month_str, year = match.groups()
        month_num = _MONTHS.get(month_str.lower())
        if month_num:
            return date(int(year), month_num, int(day))
    pattern2 = r"mse-daily-(\d{2})-(\d{2})-(\d{4})\.pdf"
    match = re.search(pattern2, filename)
    if match:
        day, month, year = match.groups()
        return date(int(year), int(month), int(day))
    pattern3 = r"mse-daily-(\d{4})-(\d{2})-(\d{2})\.pdf"
    match = re.search(pattern3, filename)
    if match:
        year, month, day = match.groups()
        return date(int(year), int(month), int(day))
    extracted_date = _parse_date_str(filename)
    if extracted_date:
        return extracted_date
    return None


def _parse_time_str(s: str):
    s = _norm_text(s)
    m = re.search(r"(?i)\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)\b", s)
    if m:
        hh, mm, ss, ap = m.groups()
        hh, mm, ss = int(hh), int(mm), int(ss or 0)
        ap = ap.lower()
        if hh == 12:
            hh = 0
        if ap == "pm":
            hh += 12
        try:
            return time(hh, mm, ss)
        except ValueError:
            return None
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))\b", s)
    if m:
        hh, mm, ss = map(int, m.groups())
        try:
            return time(hh, mm, ss)
        except ValueError:
            return None
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", s)
    if m:
        hh, mm = map(int, m.groups())
        try:
            return time(hh, mm)
        except ValueError:
            return None
    return None


def extract_print_date_time(
    pdf_path: str | Path, search_pages: int = 2, day_first: bool = True
):
    pdf_path = Path(pdf_path)
    raw_date_snip = raw_time_snip = None
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        n = min(max(search_pages, 1), len(pdf.pages))
        page_texts = []
        for i in range(n):
            page_texts.append(pdf.pages[i].extract_text() or "")
        text = "\n".join(page_texts)
    m = re.search(r"(?is)Print\s*Date\s*:?\s*([^\n\r]+)", text)
    if m:
        raw_date_snip = m.group(1)
    m = re.search(r"(?is)Print\s*Time\s*:?\s*([^\n\r]+)", text)
    if m:
        raw_time_snip = m.group(1)
    d = _parse_date_str(raw_date_snip) if raw_date_snip else _parse_date_str(text)
    t = _parse_time_str(raw_time_snip) if raw_time_snip else _parse_time_str(text)
    return {
        "date": d,
        "time": t,
        "raw_date": (raw_date_snip or None),
        "raw_time": (raw_time_snip or None),
    }


def to_numeric_clean(val):
    if val is None:
        return np.nan
    val = str(val).strip()
    if val.lower() == "none" or val == "":
        return np.nan
    if val.startswith("(") and val.endswith(")"):
        val = "-" + val[1:-1]
    val = val.replace(",", "")
    try:
        return float(val)
    except ValueError:
        return np.nan


def clean_cell(x):
    if x is None:
        return None
    x = re.sub(r"\s+", " ", str(x)).strip()
    x = x.replace("–", "-").replace("—", "-")
    return x if x else None


def is_numericish(s: Optional[str]) -> bool:
    if s is None:
        return False
    s = str(s).strip().replace(",", "")
    return bool(re.fullmatch(r"[-+]?(\d+(\.\d+)?|\.\d+)(%?)", s))


def is_header_like(row: list) -> bool:
    cells = [c for c in row if c is not None and str(c).strip() != ""]
    if not cells:
        return False
    num_numeric = sum(1 for c in cells if is_numericish(c))
    num_alpha = sum(1 for c in cells if re.search(r"[A-Za-z]", str(c)))
    return (num_alpha >= max(1, len(cells) // 4)) and (num_numeric / len(cells) <= 0.5)


def is_summary_row(row: list) -> bool:
    """Identify summary rows like 'Domestic & Foreign – weighted average'."""
    cells = [clean_cell(c) for c in row if c is not None]
    return any(
        re.search(r"(?i)weighted\s*average|total|domestic\s*&\s*foreign", str(c))
        for c in cells
    )


def normalize_to_width(rows: list[list], width: int) -> list[list]:
    out = []
    for r in rows:
        r = list(r)
        if len(r) < width:
            r = r + [None] * (width - len(r))
        elif len(r) > width:
            r = r[:width]
        out.append(r)
    return out


def extract_counters_from_text(page_text: str, expected_counters: set) -> List[str]:
    """Fallback method to extract counters from raw page text."""
    counters = []
    for counter in expected_counters:
        if re.search(rf"\b{counter}\b", page_text, re.IGNORECASE):
            counters.append(counter)
    return counters


def extract_first_table(
    pdf_path: str | Path,
    out_dir: Optional[str | Path] = None,
    header: Optional[List[str]] = None,
    skip_header_rows: int = 0,
    auto_skip_header_like: bool = True,
    logs_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Extract the first table. If header is provided, we will:
      - optionally auto-skip any header-like rows at the top
      - then force DataFrame columns to header
    """
    with pdfplumber.open(pdf_path) as pdf:
        # Determine period based on print_date
        info = extract_print_date_time(pdf_path)
        print_date = info["date"]
        if print_date is None:
            logger.warning(f"No print date found in {pdf_path.name}")
            return pd.DataFrame()

        if print_date >= date(2021, 1, 1):
            period = "2021-2025"
            column_mapping = None
            expected_row_count = 16
        elif print_date >= date(2020, 1, 1):
            period = "2020"
            column_mapping = COLUMN_MAPPING_2019_2020
            expected_row_count = len(COUNTER_LIST["2020"])
        elif print_date >= date(2019, 1, 1):
            period = "2019"
            column_mapping = COLUMN_MAPPING_2019_2020
            expected_row_count = len(COUNTER_LIST["2019"])
        elif print_date >= date(2018, 1, 1):
            period = "2018"
            column_mapping = COLUMN_MAPPING_2017_2018
            expected_row_count = len(COUNTER_LIST["2018"])
        else:
            period = "2017"
            column_mapping = COLUMN_MAPPING_2017_2018
            expected_row_count = len(COUNTER_LIST["2017"])

        expected_counters = set(COUNTER_LIST[period])
        expected_cols = COLS[period]
        # Extract raw text for fallback counter extraction
        page_text = "\n".join(page.extract_text() or "" for page in pdf.pages[:2])
        for page in pdf.pages:
            # Try multiple table extraction strategies
            strategies = [
                dict(
                    vertical_strategy="lines",
                    horizontal_strategy="lines",
                    snap_tolerance=10,
                    join_tolerance=10,
                    edge_min_length=3,
                ),
                dict(
                    vertical_strategy="lines_strict", horizontal_strategy="lines_strict"
                ),
                dict(vertical_strategy="text", horizontal_strategy="text"),
            ]
            tables = []
            for ts in strategies:
                try:
                    t = page.extract_tables(table_settings=ts) or []
                    for raw in t:
                        if raw and len(raw) >= 2 and max(len(r) for r in raw) >= 2:
                            tables.append(raw)
                    if tables:
                        break
                except Exception:
                    continue
            if not tables:
                continue
            # Use the first table found
            raw = tables[0]
            rows = [[clean_cell(c) for c in row] for row in raw]
            rows = [r for r in rows if any(c for c in r) and not is_summary_row(r)]
            if not rows:
                continue
            if period in ["2017", "2018"]:
                logger.debug(f"Raw table for {pdf_path.name}: {rows[:5]}")
            # Decide how many rows to skip from top if header is provided
            start_idx = 0
            if header:
                if auto_skip_header_like:
                    auto_skip = 0
                    for r in rows:
                        if is_header_like(r):
                            auto_skip += 1
                        else:
                            break
                    start_idx = auto_skip
                start_idx = max(start_idx, skip_header_rows)
                cols = list(header)
            else:
                detected = rows[0]
                start_idx = 1
                cols = []
                seen = {}
                for i, name in enumerate(detected):
                    name = name or f"col_{i+1}"
                    name = re.sub(r"\s+", " ", name).strip()
                    if name in seen:
                        seen[name] += 1
                        name = f"{name}_{seen[name]}"
                    else:
                        seen[name] = 1
                    cols.append(name)
            # Build DataFrame
            data_rows = normalize_to_width(rows[start_idx:], len(cols))
            df = pd.DataFrame(data_rows, columns=cols).dropna(how="all")
            if period in ["2017", "2018"]:
                logger.debug(
                    f"DataFrame before filtering for {pdf_path.name}:\n{df.head().to_string()}"
                )
            # For 2017-2018, relax counter filtering
            if period in ["2017", "2018"]:
                # Only filter out rows with no counter value
                df = df[df["counter"].notna()]
            else:
                df = df[df["counter"].isin(expected_counters)]

            # Map columns to standard format
            if column_mapping:
                df = df.rename(columns=column_mapping)
                for col in expected_cols:
                    if col not in df.columns:
                        df[col] = np.nan
            # Ensure only expected columns are kept
            df = df[[col for col in expected_cols if col in df.columns]]
            # Generate counter_id for 2017-2018 if missing
            if period in ["2017", "2018"] and "counter_id" not in df.columns:
                df["counter_id"] = range(1, len(df) + 1)
            # Convert counter_id to integer
            if "counter_id" in df.columns:
                df["counter_id"] = pd.to_numeric(
                    df["counter_id"], errors="coerce"
                ).astype("Int64")
            # Convert to numeric where possible
            for c in df.columns:
                if c != "counter":
                    df[c] = df[c].apply(to_numeric_clean)
            # Add date and print time to df
            df["trade_date"] = print_date
            df["print_time"] = info["time"]
            if period in ["2017", "2018"]:
                logger.debug(
                    f"DataFrame after processing for {pdf_path.name}:\n{df.to_string()}"
                )
            # Create CSV file based on date
            out_csv = out_dir / f"mse-daily-{print_date}.csv" if out_dir else None
            # Run checks to ensure structural correctness
            try:
                if df.empty:
                    logger.warning(f"No data extracted from {pdf_path.name}")
                    if logs_dir:
                        logs_dir.mkdir(parents=True, exist_ok=True)
                        failed_log_path = logs_dir / "failed_extractions.log"
                        with open(failed_log_path, "a") as f:
                            f.write(f"{pdf_path.name}\n")
                        print(f"❌ Failed to extract data from {pdf_path.name}, logged to {failed_log_path}")
                    else:
                        print(f"❌ Failed to extract data from {pdf_path.name}")
                    return pd.DataFrame()

                actual_counters = set(df["counter"].dropna().unique())

                # Fallback: Extract counters from text if none found
                if not actual_counters and period in ["2017", "2018"]:
                    actual_counters = set(
                        extract_counters_from_text(page_text, expected_counters)
                    )
                    logger.warning(
                        f"Fallback: Extracted counters from text for {pdf_path.name}: {actual_counters}"
                    )

                if actual_counters != expected_counters:
                    missing_counters = expected_counters - actual_counters
                    extra_counters = actual_counters - expected_counters
                    logger.warning(
                        f"Counter mismatch in {pdf_path.name}: "
                        f"Missing {missing_counters}, Extra {extra_counters}"
                    )

                if len(df) != expected_row_count:
                    logger.warning(
                        f"Row count mismatch in {pdf_p"2021-2025"ath.name}: "
                        f"Expected {expected_row_count} rows, Got {len(df)}"
                    )

                missing_cols = [col for col in expected_cols if col not in df.columns]
                if missing_cols:
                    logger.warning(
                        f"Missing columns in {pdf_path.name}: {missing_cols}"
                    )
            except Exception as e:
                logger.error(f"Structural check failed for {pdf_path.name}: {e}")
                return pd.DataFrame()
            if out_dir:
                df.to_csv(out_csv, index=False)
                print(f"✅ First table extracted and saved to {out_csv}")
                return out_csv
            return df
    logger.warning(f"No table found in {pdf_path.name}")
    return pd.DataFrame()


def get_most_recent_mse_report(directory_path):
    """
    Find the most recent MSE daily report PDF in a directory.
    """
    try:
        directory = Path(directory_path)
        if not directory.exists():
            return None
        date_patterns = [
            r"(?:Daily|Daily_Report)(\d{1,2})([A-Za-z]+)_(\d{4})\.pdf",
            r"mse-daily-(\d{2})-(\d{2})-(\d{4})\.pdf",
            r"mse-daily-(\d{4})-(\d{2})-(\d{2})\.pdf",
        ]
        pdf_files = []
        for pdf_file in directory.glob("*.pdf"):
            file_date = None
            for pattern in date_patterns:
                match = re.search(pattern, pdf_file.name)
                if match:
                    groups = match.groups()
                    try:
                        if pattern.startswith(r"(?:Daily|Daily_Report)"):
                            day, month_str, year = groups
                            month_num = _MONTHS.get(month_str.lower())
                            if month_num:
                                file_date = datetime(int(year), month_num, int(day))
                        elif pattern.startswith(r"mse-daily-(\d{2})"):
                            day, month, year = groups
                            file_date = datetime(int(year), int(month), int(day))
                        elif pattern.startswith(r"mse-daily-(\d{4})"):
                            year, month, day = groups
                            file_date = datetime(int(year), int(month), int(day))
                        break
                    except ValueError:
                        continue
            if file_date:
                pdf_files.append((file_date, pdf_file))
        if not pdf_files:
            return None
        pdf_files.sort(key=lambda x: x[0], reverse=True)
        return str(pdf_files[0][1])
    except Exception as e:
        logger.error(f"Error finding most recent MSE report: {e}")
        return None


def process_multiple_pdfs(
    input_dir: Path,
    out_dir: Path,
    start_date: date,
    cols: List[str],
    logs_dir: Optional[str | Path] = None,
) -> List[Optional[Path]]:
    not_processed = []
    for pdf_path in input_dir.glob("*.pdf"):
        try:
            file_date = extract_date_from_filename(pdf_path)
            if not file_date:
                print(f"⚠️ Skipping (no date in filename): {pdf_path.name}")
                continue
            if file_date >= start_date:
                print(f"Processing {pdf_path.name} dated {file_date}")
                output_file = extract_first_table(
                    pdf_path=pdf_path,
                    out_dir=out_dir,
                    header=cols,
                    skip_header_rows=1,
                    auto_skip_header_like=True,
                    logs_dir= logs_dir,
                )
                if output_file:
                    print(f"✅ Successfully Processed {pdf_path.name} -> {output_file}")
                else:
                    print(f"❌ Failed to process {pdf_path.name}")
                    not_processed.append(pdf_path.name)
                    continue
        except Exception as e:
            print(f"❌ Error processing {pdf_path.name}: {e}")
            output_file = None
    # Write to file unprocessed PDF filenames
    if not_processed:
        logs_dir.mkdir(parents=True, exist_ok=True)
        # log_file = Path(logs_dir) / "unprocessed_daily_pdfs.txt"
        log_file = logs_dir / "unprocessed_daily_pdfs.txt"
        with open(log_file, "w") as f:
            for fname in not_processed:
                f.write(f"{fname}\n")
        print(f"Unprocessed PDF filenames written to {log_file}")
    # return not_processed


def process_latest_report(
    input_dir: Path, out_dir: Path, cols: List[str]
) -> List[Optional[Path]]:
    """
    Process the most recent MSE report PDF in input_dir, saving extracted data to out_dir as CSV.
    """
    pdf_path = get_most_recent_mse_report(input_dir)
    logger.info(f"Most recent report: {pdf_path}")
    if not Path(pdf_path).exists():
        logger.error(f"File {pdf_path} not found")
        sys.exit(1)
    logger.info(f"Extracting data from: {pdf_path}")
    output_file = extract_first_table(
        pdf_path=pdf_path,
        out_dir=out_dir,
        header=cols,
        skip_header_rows=1,
        auto_skip_header_like=True,
    )
    if output_file:
        print(f"✅ Data extraction completed successfully")
        print(f"📁 CSV file ready for inspection: {output_file}")
        print(f"\n💡 Next steps:")
        print(f"   1. Review the CSV file: {output_file}")
        print(f"   2. Load data: python mse_data_loader.py {output_file}")
    else:
        print("❌ Failed to save data to CSV")
        sys.exit(1)
    # return [output_file]


def merge_csv_into_master(data_dir: Path, master_csv: Path, cols: List[str]):
    all_files = sorted(data_dir.glob("mse-daily-*.csv"))
    if not all_files:
        print(f"No CSV files found in {data_dir}")
        return
    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            df_list.append(df)
            print(f"Loaded {file} with {len(df)} records")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    if not df_list:
        print("No valid data to combine")
        return
    master_df = pd.concat(df_list, ignore_index=True)
    master_df = master_df[cols + ["trade_date", "print_time"]]
    master_df.drop_duplicates(
        subset=["counter_id", "trade_date"], keep="last", inplace=True
    )
    master_df.sort_values(
        by=["trade_date", "counter_id"], ascending=[False, True], inplace=True
    )
    master_df.to_csv(master_csv, index=False)
    print(f"✅ Master CSV created at {master_csv} with {len(master_df)} unique records")


def main(process_latest=True, start_date_str="2017-01-01"):
    script_dir = Path(__file__).parent.parent
    DIR_DATA = script_dir.parent / "data"
    DIR_REPORTS_PDF = DIR_DATA / "mse-daily-reports"
    DIR_REPORTS_CSV = DIR_DATA / "mse-daily-data"
    DIR_LOGS = script_dir / "logs/unprocessed_daily_pdfs"

    cols = COLS["2021-2025"]
    if process_latest:
        process_latest_report(DIR_REPORTS_PDF, DIR_REPORTS_CSV,cols)
    else:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        print(f"Processing all reports from {start_date} onwards...")
        process_multiple_pdfs(
            DIR_REPORTS_PDF, DIR_REPORTS_CSV, start_date, cols, DIR_LOGS
        )

if __name__ == "__main__":
    PROCESS_LATEST = False
    main(process_latest=PROCESS_LATEST)