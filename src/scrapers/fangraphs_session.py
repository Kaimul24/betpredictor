import json
import os
from pathlib import Path

import scrapy
from scrapy.exceptions import CloseSpider

try:
    from src.config import PROJECT_ROOT
except ImportError:
    from config import PROJECT_ROOT


FANGRAPHS_CONTEXT = "fangraphs"
FANGRAPHS_HOME = "https://www.fangraphs.com/"
FANGRAPHS_STATE_ENV = "FANGRAPHS_STORAGE_STATE"
FANGRAPHS_UA_ENV = "FANGRAPHS_USER_AGENT"
DEFAULT_FANGRAPHS_STATE_PATH = PROJECT_ROOT / "src" / "scrapers" / ".fangraphs_state.json"

SESSION_REFRESH_COMMAND = "uv run python -m src.tools.refresh_fangraphs_session"

FANGRAPHS_SPIDER_SETTINGS = {
    "CONCURRENT_REQUESTS": 2,
    "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
    "DOWNLOAD_DELAY": 1.0,
    "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,
    "PLAYWRIGHT_MAX_PAGES_PER_CONTEXT": 2,
}


def fangraphs_storage_state_path() -> Path:
    configured_path = os.getenv(FANGRAPHS_STATE_ENV)
    if configured_path:
        return Path(configured_path).expanduser()
    return DEFAULT_FANGRAPHS_STATE_PATH


def fangraphs_user_agent() -> str | None:
    return os.getenv(FANGRAPHS_UA_ENV)


def fangraphs_context_kwargs() -> dict:
    context = {
        "ignore_https_errors": True,
        "extra_http_headers": {
            "Accept-Language": "en-US,en;q=0.9",
        },
    }

    user_agent = fangraphs_user_agent()
    if user_agent:
        context["user_agent"] = user_agent

    state_path = fangraphs_storage_state_path()
    if _storage_state_is_readable(state_path):
        context["storage_state"] = str(state_path)

    return context


def fangraphs_headers(settings) -> dict:
    headers = {
        **settings.getdict("DEFAULT_REQUEST_HEADERS"),
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": FANGRAPHS_HOME,
        "Origin": FANGRAPHS_HOME.rstrip("/"),
        "Sec-Fetch-Mode": "cors",
    }
    user_agent = fangraphs_user_agent()
    if user_agent:
        headers["User-Agent"] = user_agent
    return headers


def fangraphs_meta() -> dict:
    return {
        "playwright": True,
        "playwright_context": FANGRAPHS_CONTEXT,
        "playwright_page_goto_kwargs": {
            "wait_until": "domcontentloaded",
            "timeout": 45_000,
        },
    }


def fangraphs_request(
    url: str,
    *,
    callback,
    cb_kwargs: dict | None = None,
    headers: dict | None = None,
    dont_filter: bool = False,
) -> scrapy.Request:
    return scrapy.Request(
        url,
        callback=callback,
        cb_kwargs=cb_kwargs or {},
        headers=headers or {},
        meta=fangraphs_meta(),
        dont_filter=dont_filter,
    )


def require_fangraphs_storage_state(spider) -> Path:
    state_path = fangraphs_storage_state_path()
    if _storage_state_is_readable(state_path):
        return state_path

    if state_path.exists():
        spider.logger.error(
            "FanGraphs session state at %s is not valid JSON. Run `%s`, solve "
            "the browser verification, then rerun this spider.",
            state_path,
            SESSION_REFRESH_COMMAND,
        )
        raise CloseSpider("invalid_fangraphs_session_state")

    spider.logger.error(
        "FanGraphs session state not found at %s. Run `%s`, solve the browser "
        "verification, then rerun this spider.",
        state_path,
        SESSION_REFRESH_COMMAND,
    )
    raise CloseSpider("missing_fangraphs_session_state")


def load_fangraphs_json(response, logger=None):
    if response.status == 403:
        _close_for_expired_session(response.url, logger)

    text = _extract_response_text(response)
    if not text:
        _close_for_expired_session(response.url, logger)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        _close_for_expired_session(response.url, logger)


def _extract_response_text(response) -> str:
    raw = response.text.strip()
    if raw.startswith("{") or raw.startswith("["):
        return raw

    pre_text = response.xpath("string(//pre)").get()
    if pre_text:
        text = pre_text.strip()
        if text.startswith("{") or text.startswith("["):
            return text

    return ""


def _close_for_expired_session(url: str, logger=None) -> None:
    message = (
        f"FanGraphs session expired or blocked while fetching {url}. "
        f"Run `{SESSION_REFRESH_COMMAND}`, solve the browser verification, "
        "then rerun the spider."
    )
    if logger:
        logger.error(message)
    raise CloseSpider("expired_fangraphs_session_state")


def _storage_state_is_readable(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return True
