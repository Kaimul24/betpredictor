import argparse
import asyncio
import json
import os

from dotenv import load_dotenv
from playwright.async_api import async_playwright

from src.config import DATES
from src.scrapers.fangraphs_session import fangraphs_storage_state_path, fangraphs_user_agent


def _default_validation_url() -> str:
    year = max(DATES.keys())

    league_avg_url = os.getenv("LG_AVG_URL")
    if league_avg_url:
        return league_avg_url.format("bat", year, year)

    pitchers_url = os.getenv("PITCHERS_URL")
    if pitchers_url:
        return pitchers_url.format(year, year)

    raise RuntimeError("Set PITCHERS_URL or LG_AVG_URL in .env before refreshing FanGraphs state.")


async def refresh_session(validation_url: str | None = None) -> None:
    state_path = fangraphs_storage_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    url = validation_url or _default_validation_url()

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=False,
            args=["--ignore-certificate-errors", "--ignore-ssl-errors"],
        )

        context_kwargs = {
            "ignore_https_errors": True,
            "extra_http_headers": {
                "Accept-Language": "en-US,en;q=0.9",
            },
        }
        user_agent = fangraphs_user_agent()
        if user_agent:
            context_kwargs["user_agent"] = user_agent

        if state_path.exists():
            try:
                json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                print(f"Ignoring invalid existing FanGraphs state at {state_path}")
            else:
                context_kwargs["storage_state"] = str(state_path)

        context = await browser.new_context(**context_kwargs)
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=45_000)

        print("Complete the FanGraphs browser verification on the API page in Chromium.")
        print("Wait until the API URL shows JSON, then return here.")
        await asyncio.to_thread(input, "Press Enter to validate and save the session...")

        for attempt in range(2):
            response = await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            if response is not None and response.status != 403 and await _page_has_json(page):
                break

            if attempt == 0:
                print("FanGraphs still did not return JSON.")
                print("Complete any remaining browser verification on the API page.")
                await asyncio.to_thread(input, "Press Enter to retry validation...")
        else:
            await browser.close()
            raise RuntimeError(
                "FanGraphs validation did not return JSON. The browser session may still be blocked."
            )

        await context.storage_state(path=str(state_path))
        await browser.close()
        print(f"Saved FanGraphs storage state to {state_path}")


async def _page_has_json(page) -> bool:
    body_text = (await page.locator("body").inner_text(timeout=5_000)).strip()
    try:
        json.loads(body_text)
    except json.JSONDecodeError:
        return False
    return True


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Refresh the manually verified FanGraphs Playwright storage state."
    )
    parser.add_argument(
        "--validation-url",
        help="Optional FanGraphs API URL to validate before saving storage state.",
    )
    args = parser.parse_args()

    asyncio.run(refresh_session(args.validation_url))


if __name__ == "__main__":
    main()
