#!/usr/bin/env python3
"""Download current MOEX Blue Chip Index constituents and their full daily trading history.

Sources:
- MOEX ISS index analytics for current MOEXBC constituents.
- MOEX ISS daily candles and daily history endpoints.

The script preserves current securities and predecessor securities separately. It does not
silently splice legal instruments after redomiciliation or ticker changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ISS = "https://iss.moex.com/iss"
FALLBACK_CONSTITUENTS = [
    "LKOH", "SBER", "GAZP", "YDEX", "TATN", "T", "NVTK", "GMKN",
    "ROSN", "OZON", "X5", "PLZL", "VTBR", "SNGS", "MOEX",
]
PREDECESSORS = {
    "YDEX": ["YNDX"],
    "T": ["TCSG"],
    "X5": ["FIVE"],
}
BOARD_PRIORITY = {
    "TQBR": 0,
    "TQPI": 1,
    "TQDE": 2,
    "TQIF": 3,
    "TQTD": 4,
    "EQBR": 5,
}

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "MOEX-Blue-Chips-Research-Downloader/1.0 (+https://github.com/Jaman003/public)",
    "Accept": "application/json",
})


def request_json(url: str, params: dict[str, Any] | None = None, attempts: int = 7) -> dict[str, Any]:
    delay = 1.0
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = SESSION.get(url, params=params, timeout=60)
            if response.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"retryable HTTP {response.status_code}")
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(delay)
            delay = min(delay * 2, 30)
    raise RuntimeError(f"Failed request: {url} params={params}: {last_error}")


def block_frame(payload: dict[str, Any], block: str) -> pd.DataFrame:
    raw = payload.get(block)
    if not isinstance(raw, dict):
        return pd.DataFrame()
    columns = raw.get("columns") or []
    data = raw.get("data") or []
    return pd.DataFrame(data, columns=columns)


def cursor_values(payload: dict[str, Any], block: str) -> tuple[int, int, int] | None:
    cursor = block_frame(payload, f"{block}.cursor")
    if cursor.empty:
        return None
    cols = {str(c).upper(): c for c in cursor.columns}
    needed = [cols.get("INDEX"), cols.get("TOTAL"), cols.get("PAGESIZE")]
    if any(value is None for value in needed):
        return None
    row = cursor.iloc[0]
    return int(row[needed[0]]), int(row[needed[1]]), int(row[needed[2]])


def fetch_paginated(url: str, block: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    start = 0
    base_params = dict(params or {})
    while True:
        page_params = {**base_params, "start": start}
        payload = request_json(url, page_params)
        frame = block_frame(payload, block)
        if frame.empty:
            break
        frames.append(frame)
        cursor = cursor_values(payload, block)
        if cursor is not None:
            index, total, pagesize = cursor
            next_start = index + pagesize
            if next_start >= total or next_start <= start:
                break
            start = next_start
        else:
            page_size = len(frame)
            if page_size < 100:
                break
            start += page_size
        time.sleep(0.08)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    return result.drop_duplicates().reset_index(drop=True)


def find_ticker_column(frame: pd.DataFrame) -> str | None:
    aliases = {"ticker", "secid", "security", "tradingsymbol"}
    for column in frame.columns:
        if str(column).lower() in aliases:
            return column
    return None


def get_constituents() -> tuple[pd.DataFrame, str]:
    endpoints = [
        (f"{ISS}/statistics/engines/stock/markets/index/analytics/MOEXBC/tickers.json", "analytics"),
        (f"{ISS}/statistics/engines/stock/markets/index/analytics/MOEXBC.json", "analytics"),
    ]
    errors: list[str] = []
    for url, block in endpoints:
        try:
            payload = request_json(url, {"iss.meta": "off"})
            candidate_blocks = [block] + [key for key in payload if not key.endswith(".cursor")]
            for candidate in candidate_blocks:
                frame = block_frame(payload, candidate)
                if frame.empty:
                    continue
                ticker_col = find_ticker_column(frame)
                if ticker_col is None:
                    continue
                frame = frame.copy()
                frame["SECID"] = frame[ticker_col].astype(str).str.upper()
                frame = frame[frame["SECID"].str.fullmatch(r"[A-Z0-9-]+", na=False)]
                tickers = frame["SECID"].drop_duplicates().tolist()
                if len(tickers) >= 10:
                    frame = frame.drop_duplicates("SECID").reset_index(drop=True)
                    return frame, url
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {exc}")
    fallback = pd.DataFrame({"SECID": FALLBACK_CONSTITUENTS})
    fallback["constituent_source_note"] = "Fallback from official MOEX Blue Chip factsheet dated 2026-07-31"
    fallback["constituent_fetch_errors"] = " | ".join(errors)
    return fallback, "fallback: official MOEX July-2026 factsheet"


def security_description(secid: str) -> pd.DataFrame:
    payload = request_json(f"{ISS}/securities/{secid}.json", {"iss.meta": "off"})
    description = block_frame(payload, "description")
    if description.empty:
        return pd.DataFrame([{"SECID": secid}])
    cols = {str(c).lower(): c for c in description.columns}
    name_col = cols.get("name")
    value_col = cols.get("value")
    if name_col and value_col:
        record = {str(row[name_col]): row[value_col] for _, row in description.iterrows()}
        record["SECID"] = secid
        return pd.DataFrame([record])
    description["SECID"] = secid
    return description


def download_candles(secid: str, from_date: str, till_date: str) -> pd.DataFrame:
    columns = "begin,end,open,close,high,low,value,volume"
    frame = fetch_paginated(
        f"{ISS}/engines/stock/markets/shares/securities/{secid}/candles.json",
        "candles",
        {
            "iss.meta": "off",
            "interval": 24,
            "from": from_date,
            "till": till_date,
            "candles.columns": columns,
        },
    )
    if frame.empty:
        return frame
    frame.columns = [str(c).lower() for c in frame.columns]
    frame["date"] = pd.to_datetime(frame["begin"], errors="coerce").dt.normalize()
    frame["secid"] = secid
    numeric = ["open", "high", "low", "close", "value", "volume"]
    for column in numeric:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    ordered = ["date", "secid", "open", "high", "low", "close", "volume", "value", "begin", "end"]
    return frame[[column for column in ordered if column in frame]].sort_values("date").drop_duplicates("date")


def download_history(secid: str, from_date: str, till_date: str) -> pd.DataFrame:
    columns = (
        "SECID,BOARDID,TRADEDATE,SHORTNAME,NUMTRADES,VALUE,OPEN,LOW,HIGH,"
        "LEGALCLOSEPRICE,WAPRICE,CLOSE,VOLUME,MARKETPRICE2,MARKETPRICE3,"
        "ADMITTEDQUOTE,MP2VALTRD,MARKETPRICE3TRADESVALUE,ADMITTEDVALUE,WAVAL"
    )
    frame = fetch_paginated(
        f"{ISS}/history/engines/stock/markets/shares/securities/{secid}.json",
        "history",
        {
            "iss.meta": "off",
            "from": from_date,
            "till": till_date,
            "history.columns": columns,
        },
    )
    if frame.empty:
        return frame
    frame.columns = [str(c).lower() for c in frame.columns]
    frame["tradedate"] = pd.to_datetime(frame["tradedate"], errors="coerce").dt.normalize()
    numeric = [
        "numtrades", "value", "open", "low", "high", "legalcloseprice", "waprice",
        "close", "volume", "marketprice2", "marketprice3", "admittedquote",
        "mp2valtrd", "marketprice3tradesvalue", "admittedvalue", "waval",
    ]
    for column in numeric:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["tradedate", "boardid"]).reset_index(drop=True)


def choose_primary_history(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    frame = history.copy()
    frame["board_priority"] = frame["boardid"].map(BOARD_PRIORITY).fillna(99)
    frame["value_rank"] = frame["value"].fillna(-1)
    frame = frame.sort_values(
        ["tradedate", "board_priority", "value_rank"],
        ascending=[True, True, False],
    )
    return frame.drop_duplicates("tradedate", keep="first").drop(columns=["board_priority", "value_rank"])


def merge_normalized(secid: str, candles: pd.DataFrame, primary_history: pd.DataFrame) -> pd.DataFrame:
    if candles.empty and primary_history.empty:
        return pd.DataFrame()
    if candles.empty:
        result = primary_history.rename(columns={"tradedate": "date"}).copy()
        result["secid"] = secid
        return result
    result = candles.copy()
    if not primary_history.empty:
        selected = primary_history.rename(columns={"tradedate": "date"}).copy()
        keep = [
            "date", "boardid", "shortname", "numtrades", "value", "volume", "close",
            "waprice", "legalcloseprice", "marketprice2", "marketprice3",
        ]
        selected = selected[[c for c in keep if c in selected]].copy()
        selected = selected.rename(
            columns={
                "value": "history_value",
                "volume": "history_volume",
                "close": "history_close",
            }
        )
        result = result.merge(selected, on="date", how="left")
    result["return"] = result["close"].pct_change()
    result["log_return"] = (result["close"] / result["close"].shift(1)).map(
        lambda value: math.log(value) if pd.notna(value) and value > 0 else float("nan")
    )
    result["currency"] = "RUB"
    result["source"] = "MOEX ISS"
    result["downloaded_at_utc"] = datetime.now(timezone.utc).isoformat()
    return result.sort_values("date").reset_index(drop=True)


def quality_row(secid: str, frame: pd.DataFrame, role: str) -> dict[str, Any]:
    if frame.empty:
        return {
            "secid": secid, "role": role, "rows": 0, "first_date": None, "last_date": None,
            "duplicate_dates": None, "null_ohlc": None, "invalid_ohlc": None,
            "zero_volume": None, "large_gap_count": None, "max_abs_return": None,
            "status": "NO_DATA",
        }
    duplicate_dates = int(frame["date"].duplicated().sum())
    ohlc = [c for c in ["open", "high", "low", "close"] if c in frame]
    null_ohlc = int(frame[ohlc].isna().any(axis=1).sum()) if ohlc else None
    invalid = 0
    if set(["open", "high", "low", "close"]).issubset(frame.columns):
        invalid_mask = (
            (frame["low"] > frame[["open", "close"]].min(axis=1))
            | (frame["high"] < frame[["open", "close"]].max(axis=1))
            | (frame["low"] > frame["high"])
        )
        invalid = int(invalid_mask.fillna(False).sum())
    zero_volume = int((frame["volume"].fillna(0) <= 0).sum()) if "volume" in frame else None
    gaps = frame["date"].sort_values().diff().dt.days
    large_gap_count = int((gaps > 14).sum())
    max_abs_return = float(frame["return"].abs().max()) if "return" in frame and frame["return"].notna().any() else None
    status = "OK" if duplicate_dates == 0 and invalid == 0 and (null_ohlc or 0) == 0 else "REVIEW"
    return {
        "secid": secid,
        "role": role,
        "rows": int(len(frame)),
        "first_date": frame["date"].min().date().isoformat(),
        "last_date": frame["date"].max().date().isoformat(),
        "duplicate_dates": duplicate_dates,
        "null_ohlc": null_ohlc,
        "invalid_ohlc": invalid,
        "zero_volume": zero_volume,
        "large_gap_count": large_gap_count,
        "max_abs_return": max_abs_return,
        "status": status,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="output")
    parser.add_argument("--from-date", default="1990-01-01")
    parser.add_argument("--till-date", default=date.today().isoformat())
    args = parser.parse_args()

    root = Path(args.out).resolve()
    if root.exists():
        shutil.rmtree(root)
    for directory in [
        root / "raw" / "candles",
        root / "raw" / "history",
        root / "normalized" / "current",
        root / "normalized" / "predecessors",
        root / "metadata",
        root / "quality",
        root / "panels",
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    constituents, constituent_source = get_constituents()
    # Keep only the verified 15-index universe if ISS returns extra analytics rows.
    if len(constituents) != 15:
        official_order = FALLBACK_CONSTITUENTS
        available = set(constituents.get("SECID", pd.Series(dtype=str)).astype(str))
        rows = []
        for ticker in official_order:
            rows.append({"SECID": ticker, "from_iss": ticker in available})
        constituents = pd.DataFrame(rows)
    constituents["index_code"] = "MOEXBC"
    constituents["as_of_download"] = args.till_date
    constituents["source_url"] = constituent_source
    constituents.to_csv(root / "metadata" / "current_moexbc_constituents.csv", index=False)

    predecessor_records = []
    for current, predecessors in PREDECESSORS.items():
        for predecessor in predecessors:
            predecessor_records.append({
                "current_secid": current,
                "predecessor_secid": predecessor,
                "relationship": "ECONOMIC_PREDECESSOR_NOT_AUTOMATICALLY_SPLICED",
            })
    pd.DataFrame(predecessor_records).to_csv(root / "metadata" / "predecessor_map.csv", index=False)

    metadata_frames: list[pd.DataFrame] = []
    quality: list[dict[str, Any]] = []
    current_frames: list[pd.DataFrame] = []
    predecessor_frames: list[pd.DataFrame] = []
    download_errors: list[dict[str, str]] = []

    download_plan = [(ticker, "CURRENT_CONSTITUENT", ticker) for ticker in constituents["SECID"].tolist()]
    for current, predecessors in PREDECESSORS.items():
        for predecessor in predecessors:
            download_plan.append((predecessor, "PREDECESSOR", current))

    for secid, role, linked_current in download_plan:
        print(f"Downloading {secid} ({role})", flush=True)
        try:
            description = security_description(secid)
            description["role"] = role
            description["linked_current_secid"] = linked_current
            metadata_frames.append(description)
        except Exception as exc:  # noqa: BLE001
            download_errors.append({"secid": secid, "stage": "description", "error": str(exc)})

        try:
            candles = download_candles(secid, args.from_date, args.till_date)
            candles.to_csv(root / "raw" / "candles" / f"{secid}.csv", index=False)
        except Exception as exc:  # noqa: BLE001
            download_errors.append({"secid": secid, "stage": "candles", "error": str(exc)})
            candles = pd.DataFrame()

        try:
            history = download_history(secid, args.from_date, args.till_date)
            history.to_csv(root / "raw" / "history" / f"{secid}.csv", index=False)
        except Exception as exc:  # noqa: BLE001
            download_errors.append({"secid": secid, "stage": "history", "error": str(exc)})
            history = pd.DataFrame()

        primary = choose_primary_history(history)
        normalized = merge_normalized(secid, candles, primary)
        if role == "CURRENT_CONSTITUENT":
            output_path = root / "normalized" / "current" / f"{secid}.csv"
            current_frames.append(normalized)
        else:
            output_path = root / "normalized" / "predecessors" / f"{secid}.csv"
            normalized["linked_current_secid"] = linked_current
            predecessor_frames.append(normalized)
        normalized.to_csv(output_path, index=False)
        quality.append(quality_row(secid, normalized, role))

    if metadata_frames:
        pd.concat(metadata_frames, ignore_index=True, sort=False).to_csv(
            root / "metadata" / "security_descriptions.csv", index=False
        )
    pd.DataFrame(quality).to_csv(root / "quality" / "quality_report.csv", index=False)
    pd.DataFrame(download_errors).to_csv(root / "quality" / "download_errors.csv", index=False)

    nonempty_current = [frame for frame in current_frames if not frame.empty]
    if nonempty_current:
        panel = pd.concat(nonempty_current, ignore_index=True, sort=False)
        panel = panel.sort_values(["date", "secid"]).reset_index(drop=True)
        panel.to_csv(root / "panels" / "blue_chips_daily_panel.csv.gz", index=False, compression="gzip")
        close_panel = panel.pivot(index="date", columns="secid", values="close").sort_index()
        volume_panel = panel.pivot(index="date", columns="secid", values="volume").sort_index()
        close_panel.to_csv(root / "panels" / "close_panel.csv.gz", compression="gzip")
        volume_panel.to_csv(root / "panels" / "volume_panel.csv.gz", compression="gzip")

    manifest_rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            manifest_rows.append({
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    pd.DataFrame(manifest_rows).to_csv(root / "metadata" / "file_manifest.csv", index=False)

    readme = f"""# MOEX Blue Chips Daily Trading Data\n\nDownloaded from official MOEX ISS on {datetime.now(timezone.utc).isoformat()}.\n\n- Index: MOEXBC\n- Current securities: 15\n- Date request: {args.from_date} through {args.till_date}\n- Constituents source: {constituent_source}\n- Daily candles and board-aware daily history are preserved separately.\n- YNDX/YDEX, TCSG/T and FIVE/X5 are not automatically spliced; predecessor files are separate.\n- Prices are raw exchange prices. Corporate-action and dividend total-return adjustments are not applied.\n\nThe official MOEX Blue Chip factsheet dated 2026-07-31 states that the index contains the 15 most liquid Russian issuers.\n"""
    (root / "README.md").write_text(readme, encoding="utf-8")

    archive_path = root.parent / "MOEX_Blue_Chips_Trading_Data.zip"
    if archive_path.exists():
        archive_path.unlink()
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(Path("MOEX_Blue_Chips_Trading_Data") / path.relative_to(root)))
    print(f"Created {archive_path}", flush=True)


if __name__ == "__main__":
    main()
