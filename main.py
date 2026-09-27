"""Telegram bot that scans XAU/USD and reports trading signals."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify
from telegram import Bot

API_URL = "https://api.twelvedata.com/time_series"
SYMBOL = "XAU/USD"
INTERVAL = "1h"
OUTPUT_SIZE = 200
SCAN_INTERVAL_SECONDS = 5 * 60
REQUEST_TIMEOUT_SECONDS = 30
BB_NEAR_FRACTION = 0.10
RETRY_DELAY_SECONDS = 30

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    twelve_data_api_key: str
    telegram_bot_token: str
    telegram_chat_id: str

    @classmethod
    def from_environment(cls) -> Settings:
        values = {
            "TWELVE_DATA_API_KEY": os.getenv("TWELVE_DATA_API_KEY", "").strip(),
            "TELEGRAM_BOT_TOKEN": os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            "TELEGRAM_CHAT_ID": os.getenv("TELEGRAM_CHAT_ID", "").strip(),
        }
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise RuntimeError(
                "Missing required environment variables: " + ", ".join(missing)
            )
        return cls(
            twelve_data_api_key=values["TWELVE_DATA_API_KEY"],
            telegram_bot_token=values["TELEGRAM_BOT_TOKEN"],
            telegram_chat_id=values["TELEGRAM_CHAT_ID"],
        )


def create_keep_alive_app() -> Flask:
    """Create the small HTTP app used to keep the process reachable."""
    app = Flask(__name__)

    @app.get("/")
    def status() -> tuple[str, int]:
        return "Gold Bot is running.\n", 200

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "ok", "service": "goldbot"})

    return app


def run_keep_alive_server() -> None:
    """Run Flask without enabling its development reloader."""
    app = create_keep_alive_app()
    app.run(host="0.0.0.0", port=8080, use_reloader=False)


def fetch_candles(api_key: str) -> pd.DataFrame:
    """Fetch and normalize the latest 200 hourly candles from Twelve Data."""
    response = requests.get(
        API_URL,
        params={
            "symbol": SYMBOL,
            "interval": INTERVAL,
            "outputsize": OUTPUT_SIZE,
            "apikey": api_key,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") == "error":
        message = payload.get("message", "Twelve Data returned an error")
        raise RuntimeError(message)

    values = payload.get("values")
    if not isinstance(values, list) or not values:
        raise RuntimeError("Twelve Data returned no candle values")

    candles = pd.DataFrame(values)
    required_columns = {"datetime", "close"}
    if not required_columns.issubset(candles.columns):
        raise RuntimeError("Twelve Data response is missing candle fields")

    candles["datetime"] = pd.to_datetime(candles["datetime"], utc=True)

    for column in ("open", "high", "low", "close", "volume"):
        if column in candles.columns:
            candles[column] = pd.to_numeric(
                candles[column],
                errors="coerce",
            )

    candles = candles.dropna(subset=["datetime", "close"])
    candles = candles.sort_values("datetime").reset_index(drop=True)

    if candles.empty:
        raise RuntimeError("Twelve Data returned no valid candles")

    return candles


def calculate_indicators(candles: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, EMA, and Bollinger Band columns to a candle DataFrame."""
    result = candles.copy()
    close = result["close"]

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    average_gain = gains.rolling(
        window=14,
        min_periods=14,
    ).mean()

    average_loss = losses.rolling(
        window=14,
        min_periods=14,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)

    result["rsi"] = 100 - (100 / (1 + relative_strength))

    result.loc[
        (average_loss == 0) & (average_gain > 0),
        "rsi",
    ] = 100

    result.loc[
        (average_loss == 0) & (average_gain == 0),
        "rsi",
    ] = 50

    result["ema9"] = close.ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()

    result["ema21"] = close.ewm(
        span=21,
        adjust=False,
        min_periods=21,
    ).mean()

    middle_band = close.rolling(
        window=20,
        min_periods=20,
    ).mean()

    standard_deviation = close.rolling(
        window=20,
        min_periods=20,
    ).std()

    result["bb_middle"] = middle_band
    result["bb_upper"] = middle_band + (2 * standard_deviation)
    result["bb_lower"] = middle_band - (2 * standard_deviation)

    return result


def evaluate_signal(
    candles: pd.DataFrame,
) -> tuple[str, dict[str, float]]:
    """Evaluate the latest candle and return BUY, SELL, or HOLD."""
    indicators = calculate_indicators(candles)
    latest = indicators.iloc[-1]

    values = {
        "price": float(latest["close"]),
        "rsi": float(latest["rsi"]),
        "ema9": float(latest["ema9"]),
        "ema21": float(latest["ema21"]),
        "bb_upper": float(latest["bb_upper"]),
        "bb_lower": float(latest["bb_lower"]),
    }

    if any(not np.isfinite(value) for value in values.values()):
        return "HOLD", values

    band_width = values["bb_upper"] - values["bb_lower"]

    near_lower_band = values["price"] <= (
        values["bb_lower"] + (band_width * BB_NEAR_FRACTION)
    )

    near_upper_band = values["price"] >= (
        values["bb_upper"] - (band_width * BB_NEAR_FRACTION)
    )

    if (
        values["rsi"] < 30
        and values["ema9"] > values["ema21"]
        and near_lower_band
    ):
        return "BUY", values

    if (
        values["rsi"] > 70
        and values["ema9"] < values["ema21"]
        and near_upper_band
    ):
        return "SELL", values

    return "HOLD", values


def format_signal_message(
    signal: str,
    values: dict[str, float],
) -> str:
    """Build the Telegram alert for a BUY or SELL signal."""
    return (
        f"Gold Bot signal: {signal}\n"
        f"Symbol: {SYMBOL}\n"
        f"Interval: {INTERVAL}\n"
        f"Price: {values['price']:.2f}\n"
        f"RSI: {values['rsi']:.2f}\n"
        f"EMA9: {values['ema9']:.2f}\n"
        f"EMA21: {values['ema21']:.2f}"
    )


async def scan_once(
    bot: Bot,
    settings: Settings,
) -> None:
    """Run one scan and notify Telegram only for BUY or SELL."""
    try:
        candles = await asyncio.to_thread(
            fetch_candles,
            settings.twelve_data_api_key,
        )

        signal, values = evaluate_signal(candles)

        logger.info(
            "Scan: symbol=%s interval=%s price=%.2f rsi=%.2f "
            "ema9=%.2f ema21=%.2f signal=%s",
            SYMBOL,
            INTERVAL,
            values["price"],
            values["rsi"],
            values["ema9"],
            values["ema21"],
            signal,
        )

        if signal in {"BUY", "SELL"}:
            await bot.send_message(
                chat_id=settings.telegram_chat_id,
                text=format_signal_message(signal, values),
            )

            logger.info("Sent Telegram %s signal", signal)

    except Exception:
        logger.exception(
            "Scan failed; continuing with the next scan"
        )


async def run_bot(settings: Settings) -> None:
    """Keep the Telegram bot alive and retry connection failures."""
    while True:
        try:
            async with Bot(
                token=settings.telegram_bot_token,
            ) as bot:
                try:
                    await bot.send_message(
                        chat_id=settings.telegram_chat_id,
                        text="Gold Bot online - XAU/USD 1h",
                    )

                    logger.info(
                        "Sent startup Telegram message"
                    )

                except Exception:
                    logger.exception(
                        "Could not send startup Telegram message"
                    )

                while True:
                    await scan_once(bot, settings)
                    await asyncio.sleep(
                        SCAN_INTERVAL_SECONDS
                    )

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception(
                "Telegram connection failed; retrying in %s seconds",
                RETRY_DELAY_SECONDS,
            )

            await asyncio.sleep(RETRY_DELAY_SECONDS)


def main() -> None:
    """Start the keep-alive server and Telegram polling loop."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    settings = Settings.from_environment()

    keep_alive_thread = threading.Thread(
        target=run_keep_alive_server,
        name="keep-alive",
        daemon=True,
    )

    keep_alive_thread.start()

    logger.info(
        "Keep-alive server listening on port 8080"
    )

    try:
        asyncio.run(run_bot(settings))

    except KeyboardInterrupt:
        logger.info("Gold Bot stopped")
