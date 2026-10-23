"""Exercise the real dashboard in Chromium against a running API."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright


def main(base_url: str, screenshots: Path | None) -> None:
    errors = []
    with sync_playwright() as browser_driver:
        browser = browser_driver.chromium.launch()
        context = browser.new_context(viewport={"width": 1440, "height": 1100})
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base_url)
        expect(page.get_by_text("Model connected", exact=True)).to_be_visible()
        expect(page.locator("#age-display")).to_be_hidden()
        expect(
            page.get_by_role("button", name="Download prediction as JSON")
        ).to_be_disabled()
        if screenshots:
            screenshots.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshots / "desktop-empty.png"), full_page=True)

        with page.expect_response(
            lambda response: response.url.endswith("/predict")
        ) as result:
            page.get_by_role("button", name="Estimate age", exact=True).click()
        response = result.value.json()
        expect(page.locator("#age-value")).to_have_text(f"{response['age_years']:.2f}")
        expect(page.locator("#result-badge")).to_have_text("Estimate complete")
        assert (
            page.locator("#model-fingerprint").get_attribute("title")
            == response["model_sha256"]
        )
        with page.expect_download() as download:
            page.get_by_role("button", name="Download prediction as JSON").click()
        saved = json.loads(Path(download.value.path()).read_text())
        assert saved["age_years"] == response["age_years"]
        assert saved["measurements"]["length"] == 0.455
        if screenshots:
            page.screenshot(
                path=str(screenshots / "desktop-result.png"), full_page=True
            )

        page.get_by_label("Length", exact=True).fill("0.5")
        expect(page.locator("#age-display")).to_be_hidden()
        expect(page.locator("#export-button")).to_be_disabled()
        page.get_by_role("button", name="Clear", exact=True).click()
        expect(page.get_by_label("Length", exact=True)).to_have_value("")
        page.get_by_role("button", name="Estimate age", exact=True).click()
        assert not page.locator("#prediction-form").evaluate(
            "form => form.checkValidity()"
        )
        expect(page.locator("#age-display")).to_be_hidden()
        page.get_by_role("button", name="Load example").click()
        expect(page.get_by_label("Length", exact=True)).to_have_value("0.455")

        pending = []
        page.route("**/predict", lambda route: pending.append(route))
        page.get_by_role("button", name="Estimate age", exact=True).click()
        expect(page.locator("#result-panel")).to_have_attribute("aria-busy", "true")
        expect(page.get_by_label("Length", exact=True)).to_be_disabled()
        expect(page.get_by_role("button", name="Load example")).to_be_disabled()
        expect(page.get_by_role("button", name="Clear", exact=True)).to_be_disabled()
        expect(page.locator("#predict-button")).to_be_disabled()
        assert len(pending) == 1
        pending[0].continue_()
        expect(page.locator("#result-badge")).to_have_text("Estimate complete")
        page.unroute("**/predict")

        page.route("**/predict", lambda route: route.abort())
        page.get_by_role("button", name="Estimate age", exact=True).click()
        expect(page.get_by_role("alert")).to_contain_text("couldn’t reach the model")
        expect(page.locator("#export-button")).to_be_disabled()
        page.unroute("**/predict")

        page.route(
            "**/predict",
            lambda route: route.fulfill(
                status=422,
                content_type="application/json",
                body=json.dumps(
                    {"detail": [{"loc": ["body", "length"], "msg": "Invalid"}]}
                ),
            ),
        )
        page.get_by_role("button", name="Estimate age", exact=True).click()
        expect(page.get_by_role("alert")).to_contain_text("check your measurements")
        expect(page.get_by_label("Length", exact=True)).to_have_attribute(
            "aria-invalid", "true"
        )
        page.unroute("**/predict")
        page.get_by_role("button", name="Load example").click()
        page.get_by_role("button", name="Estimate age", exact=True).click()
        expect(page.locator("#result-badge")).to_have_text("Estimate complete")

        # Negative numbers and missing values must never silently become zero.
        page.get_by_label("Height", exact=True).fill("-1")
        assert not page.locator("#prediction-form").evaluate(
            "form => form.checkValidity()"
        )
        page.get_by_role("button", name="Load example").click()

        for width in [390, 320, 768]:
            page.set_viewport_size({"width": width, "height": 844})
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            ), width
            page.get_by_role("button", name="Estimate age", exact=True).click()
            expect(page.locator("#result-badge")).to_have_text("Estimate complete")
            if screenshots:
                page.screenshot(
                    path=str(screenshots / f"mobile-{width}.png"), full_page=True
                )

        # Native radio controls and buttons remain keyboard accessible.
        page.get_by_role("radio", name="Female").check()
        page.get_by_role("button", name="Estimate age", exact=True).focus()
        page.keyboard.press("Enter")
        expect(page.locator("#result-badge")).to_have_text("Estimate complete")
        page.get_by_text("How should I interpret the result?", exact=False).click()
        expect(page.locator(".method-details")).to_have_attribute("open", "")
        page.emulate_media(reduced_motion="reduce")
        assert page.evaluate("matchMedia('(prefers-reduced-motion: reduce)').matches")
        assert not errors, errors
        browser.close()
    with urlopen(f"{base_url}/health", timeout=5) as response:
        assert response.status == 200
    print(
        "Browser checks passed: predictions, download, reset, errors, keyboard, and responsive layouts."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--screenshots", type=Path)
    args = parser.parse_args()
    main(args.base_url.rstrip("/"), args.screenshots)
