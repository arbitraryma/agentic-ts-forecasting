"""
Public CSV loading.

This helper downloads text. It does not clone a
repository or execute anything from the response.
"""

import io
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd


MAX_BYTES = 5_000_000
TIMEOUT_SECONDS = 20

BLOCKED_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "169.254.169.254",
}


def github_blob_to_raw(url):
    """Convert a public GitHub file page to a raw URL."""

    parsed = urlparse(url)
    if parsed.netloc != "github.com":
        return url

    parts = [
        part
        for part in parsed.path.split("/")
        if part
    ]

    if len(parts) < 5 or parts[2] != "blob":
        return url

    owner, repo, ref = parts[0], parts[1], parts[3]
    path = "/".join(parts[4:])
    return (
        "https://raw.githubusercontent.com/"
        f"{owner}/{repo}/{ref}/{path}"
    )


def load_public_csv_url(url, target_column):
    """
    Download a public CSV and return one numeric column.

    Only http and https URLs are accepted.
    """

    if not isinstance(url, str) or not url.strip():
        raise ValueError("A CSV URL is required.")

    raw_url = github_blob_to_raw(url.strip())
    parsed = urlparse(raw_url)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError(
            "Only http and https URLs are allowed."
        )

    host = (parsed.hostname or "").lower()
    if host in BLOCKED_HOSTS or host.endswith(".local"):
        raise ValueError(
            "That host is not allowed."
        )

    request = Request(
        raw_url,
        headers={"User-Agent": "agentic-ts-forecasting"},
    )

    with urlopen(
        request,
        timeout=TIMEOUT_SECONDS,
    ) as response:
        content_type = (
            response.headers.get("Content-Type") or ""
        ).lower()
        if (
            "html" in content_type
            and "csv" not in content_type
        ):
            raise ValueError(
                "The URL did not return CSV content."
            )

        payload = response.read(MAX_BYTES + 1)

    if len(payload) > MAX_BYTES:
        raise ValueError(
            "The CSV response is larger than 5 MB."
        )

    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "The URL did not return UTF-8 CSV text."
        ) from exc

    if "," not in text and "\n" not in text:
        raise ValueError(
            "The URL response is not a CSV table."
        )

    frame = pd.read_csv(io.StringIO(text))

    if target_column not in frame.columns:
        columns = ", ".join(map(str, frame.columns))
        raise ValueError(
            f"Target column '{target_column}' was not "
            f"found. Columns: {columns}"
        )

    return frame
