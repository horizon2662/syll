"""Chrome DevTools Protocol (CDP) verifier for browser tasks.

Reads live browser state via the CDP HTTP endpoint. The ``current_url`` /
``url_contains`` checks work dependency-free via ``/json``. DOM and
localStorage checks require evaluating against a CDP WebSocket target, which
needs a websocket client inside the sandbox — left as a clearly-marked TODO
until the Docker browser backend lands.

Spec keys:
    cdp_endpoint: str             default http://localhost:9222
    url_equals:   str             active tab URL must equal exactly
    url_contains: str             active tab URL must contain this
    # (TODO, needs CDP WebSocket eval):
    dom_exists:     str           CSS selector that must be present
    localstorage:   dict[str,str] key -> expected value
"""

from __future__ import annotations

import json
from typing import Any

from syll.sandbox.environment import Environment

from .base import Verifier, VerifierResult

_DEFAULT_ENDPOINT = "http://localhost:9222"


class CDPVerifier(Verifier):
    name = "cdp"

    async def check(self, env: Environment, spec: dict[str, Any]) -> VerifierResult:
        endpoint = spec.get("cdp_endpoint") or _DEFAULT_ENDPOINT
        if "dom_exists" in spec or "localstorage" in spec:
            return VerifierResult.fail(
                "CDP DOM/localStorage checks need a WebSocket target eval "
                "(TODO: wire when the Docker browser backend lands)",
                spec=spec,
            )
        tabs = await self._list_tabs(env, endpoint)
        if tabs is None:
            return VerifierResult.fail(
                f"could not reach CDP endpoint {endpoint}",
                cdp_endpoint=endpoint,
            )
        # Pick the first non-DevTools tab.
        urls = [
            t.get("url", "")
            for t in tabs
            if t.get("type") == "page" and not t.get("url", "").startswith("chrome://")
        ]
        active = urls[0] if urls else ""
        if "url_equals" in spec:
            if active == spec["url_equals"]:
                return VerifierResult.pass_(
                    f"url match: {active!r}", url=active
                )
            return VerifierResult.fail(
                f"url mismatch: got {active!r}, want {spec['url_equals']!r}",
                actual=active,
                expected=spec["url_equals"],
            )
        if "url_contains" in spec:
            needle = str(spec["url_contains"])
            if any(needle in u for u in urls) or needle in active:
                return VerifierResult.pass_(
                    f"url contains {needle!r}", url=active
                )
            return VerifierResult.fail(
                f"no open tab contains {needle!r}; active={active!r}",
                active=active,
                needle=needle,
            )
        # Reachable + no assertion: treat as passed gate.
        return VerifierResult.pass_(
            f"CDP reachable, {len(tabs)} tab(s)", tabs=len(tabs)
        )

    async def _list_tabs(self, env: Environment, endpoint: str) -> list[dict] | None:
        script = (
            "import json,urllib.request,sys\n"
            "try:\n"
            "    data=urllib.request.urlopen(sys.argv[1]+'/json',timeout=3).read()\n"
            "    print(data.decode())\n"
            "except Exception as e:\n"
            "    sys.exit(1)\n"
        )
        try:
            res = await env.exec(f"python3 -c '{script}' '{endpoint}'")
        except Exception:
            return None
        if res.returncode != 0:
            return None
        try:
            return json.loads(res.stdout)
        except Exception:
            return None
