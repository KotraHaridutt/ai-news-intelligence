"""
Checks that each key the app holds can do only what the app needs (feature 3, Section M #99), so a
leaked key does as little harm as possible. Reads the keys from .env (or the environment) and asks
each service, with read-only requests, what the key may do. It never prints a key, and it uses no
quota: no Gemini answer, no Clef call, no GNews search, no Redis write.

  python scripts/check_key_scopes.py            every key that is set
  python scripts/check_key_scopes.py redis      one of: gemini, cloudflare, langsmith, redis

  Gemini key      lists models (needed: works), then calls two other Google APIs with it: refused
                  with API_KEY_SERVICE_BLOCKED = the key is restricted to the APIs it was given (PASS);
                  refused only because the API is off in the project = the key itself isn't restricted (WARN)
  Cloudflare      the token is active, sees one account (this one), can read Workers AI, and can't read
                  Workers scripts, zones (DNS), R2 buckets or the AI Gateway's settings
  LangSmith       a service key (lsv2_sk_), not a personal token (lsv2_pt_), and how many workspaces it sees
  Redis           TLS (rediss://), which user the app logs in as, and whether that user may run admin
                  commands the app never uses (CONFIG, CLIENT LIST: read-only, harmless if allowed)
The M doc (implemented/features/feature-3/docs/M-agency-and-least-privilege.md §6) has the steps to fix a WARN or FAIL.
Exit code: 0 = no FAIL, 1 = a FAIL, 2 = nothing could be checked.
"""
import os
import sys
from urllib.parse import urlsplit

import httpx
from dotenv import load_dotenv

load_dotenv()

results: list[tuple[str, str, str]] = []      # (status, key, what)


def say(status: str, key: str, what: str):
    results.append((status, key, what))
    print(f"{status:<5} {key:<11} {what}")


def _error_reasons(response: httpx.Response) -> str:
    """Google's error reason(s) and status, e.g. 'API_KEY_SERVICE_BLOCKED' or 'SERVICE_DISABLED'."""
    try:
        error = response.json().get("error", {})
    except ValueError:
        return f"HTTP {response.status_code}"
    reasons = [d.get("reason") for d in error.get("details", []) if isinstance(d, dict) and d.get("reason")]
    reasons += [e.get("reason") for e in error.get("errors", []) if isinstance(e, dict) and e.get("reason")]
    return ", ".join(dict.fromkeys(r for r in reasons if r)) or error.get("status") or f"HTTP {response.status_code}"


# --- Gemini ---
# Other Google APIs a leaked key could be used on. GET requests that change nothing.
OTHER_GOOGLE_APIS = {
    "YouTube Data API": "https://www.googleapis.com/youtube/v3/videos?part=id&id=dQw4w9WgXcQ",
    "Cloud Translation API": "https://translation.googleapis.com/language/translate/v2/languages",
}


def check_gemini(client: httpx.Client) -> bool:
    key = (os.getenv("GOOGLE_API_KEY") or "").strip()
    if not key:
        return False
    headers = {"x-goog-api-key": key}
    r = client.get("https://generativelanguage.googleapis.com/v1beta/models", params={"pageSize": 1}, headers=headers)
    if r.status_code == 200:
        say("PASS", "gemini", "the key works for the Generative Language API (models.list, no quota used)")
    else:
        say("FAIL", "gemini", f"the key doesn't work for the Generative Language API: {_error_reasons(r)}")
    for name, url in OTHER_GOOGLE_APIS.items():
        r = client.get(url, headers=headers)
        reasons = _error_reasons(r)
        if r.status_code == 200:
            say("FAIL", "gemini", f"the key works for the {name}: restrict it to the Generative Language API")
        elif "API_KEY_SERVICE_BLOCKED" in reasons:
            say("PASS", "gemini", f"{name} refused: the key is restricted to other APIs ({reasons})")
        elif "ACCESS_DENIED" in reasons or "PERMISSION_DENIED" in reasons:
            # Measured 2026-11-26 on the app's AI Studio key: "Permission denied on the service"
            say("PASS", "gemini", f"{name} refused: the key has no access to it ({reasons})")
        elif "SERVICE_DISABLED" in reasons or "accessNotConfigured" in reasons:
            say("WARN", "gemini", f"{name} refused only because it's off in the key's project ({reasons}): "
                                  "the key itself has no API restriction")
        else:
            say("INFO", "gemini", f"{name}: HTTP {r.status_code} ({reasons})")
    return True


# --- Cloudflare ---
CF_API = "https://api.cloudflare.com/client/v4"


def check_cloudflare(client: httpx.Client) -> bool:
    token = (os.getenv("CLOUDFLARE_API_TOKEN") or "").strip()
    account = (os.getenv("CLOUDFLARE_ACCOUNT_ID") or "").strip()
    if not token or not account:
        return False
    headers = {"Authorization": f"Bearer {token}"}
    get = lambda path, **params: client.get(f"{CF_API}{path}", headers=headers, params=params or None)

    # A user token verifies at /user/tokens/verify, an account-owned token at /accounts/<id>/tokens/verify
    for kind, path in (("user", "/user/tokens/verify"), ("account", f"/accounts/{account}/tokens/verify")):
        r = get(path)
        if r.status_code == 200:
            result = r.json().get("result") or {}
            expires = result.get("expires_on") or "never"
            say("PASS" if result.get("status") == "active" else "FAIL", "cloudflare",
                f"{kind} token, status {result.get('status')}, expires {expires}")
            break
    else:
        say("FAIL", "cloudflare", f"the token couldn't be verified (HTTP {r.status_code})")
        return True

    r = get("/accounts", per_page=50)
    accounts = [a.get("id") for a in (r.json().get("result") or [])] if r.status_code == 200 else None
    if accounts is None:
        say("INFO", "cloudflare", f"listing accounts: HTTP {r.status_code}")
    elif not accounts:
        say("PASS", "cloudflare", "can't list accounts (no Account Settings permission)")
    elif accounts == [account]:
        say("PASS", "cloudflare", "sees exactly one account: CLOUDFLARE_ACCOUNT_ID")
    else:
        say("FAIL", "cloudflare", f"sees {len(accounts)} accounts: limit the token to CLOUDFLARE_ACCOUNT_ID")

    r = get(f"/accounts/{account}/ai/models/search", per_page=1)
    say("PASS" if r.status_code == 200 else "FAIL", "cloudflare",
        f"Workers AI read (needed): HTTP {r.status_code}" + (" (works)" if r.status_code == 200 else ""))

    # What it must NOT be able to read (a token that can't read these can't change them either). Settings
    # endpoints, not lists: Cloudflare answers some lists (workers/scripts) with 200 and an empty list to a
    # token without the permission (measured 2026-11-26), which can't be told apart from "no scripts".
    for what, path in (("Workers (account settings)", f"/accounts/{account}/workers/account-settings"),
                       ("Workers (subdomain)", f"/accounts/{account}/workers/subdomain"),
                       ("Pages projects", f"/accounts/{account}/pages/projects"),
                       ("KV namespaces", f"/accounts/{account}/storage/kv/namespaces"),
                       ("D1 databases", f"/accounts/{account}/d1/database"),
                       ("R2 buckets", f"/accounts/{account}/r2/buckets"),
                       ("account members", f"/accounts/{account}/members"),
                       ("AI Gateway settings", f"/accounts/{account}/ai-gateway/gateways")):
        r = get(path)
        if r.status_code in (401, 403):
            say("PASS", "cloudflare", f"{what}: refused (HTTP {r.status_code})")
        elif r.status_code == 200:
            say("FAIL", "cloudflare", f"{what}: readable. Remove that permission from the token")
        else:
            say("INFO", "cloudflare", f"{what}: HTTP {r.status_code}")
    r = get("/zones", per_page=5)
    zones = r.json().get("result") if r.status_code == 200 else None
    if r.status_code in (401, 403) or zones == []:
        say("PASS", "cloudflare", "zones (DNS): none visible")
    elif zones:
        say("FAIL", "cloudflare", f"zones (DNS): sees {len(zones)}+ zone(s). Remove zone permissions from the token")
    else:
        say("INFO", "cloudflare", f"zones: HTTP {r.status_code}")
    return True


# --- LangSmith ---

def check_langsmith(client: httpx.Client) -> bool:
    key = (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()
    if not key:
        return False
    if key.startswith("lsv2_sk_"):
        say("PASS", "langsmith", "a service key (lsv2_sk_): not tied to a person's account")
    elif key.startswith("lsv2_pt_"):
        say("WARN", "langsmith", "a personal access token (lsv2_pt_): it acts as you in every workspace you can open. "
                                 "Use a service key scoped to this workspace")
    else:
        say("INFO", "langsmith", "a key of an older format: replace it with a service key")
    endpoint = (os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com").rstrip("/")
    r = client.get(f"{endpoint}/api/v1/workspaces", headers={"x-api-key": key})
    workspaces = r.json() if r.status_code == 200 and isinstance(r.json(), list) else None
    if workspaces is not None:
        count = len(workspaces)
        say("PASS" if count == 1 else "WARN", "langsmith", f"sees {count} workspace{'s' if count != 1 else ''}")
    else:
        say("INFO", "langsmith", f"listing workspaces: HTTP {r.status_code}")
    # A service key made at the organization level works only with the workspace named on each request
    # (LANGSMITH_WORKSPACE_ID -> X-Tenant-Id). Without it every trace is refused 403, silently: measured
    # 2026-11-26 after the switch to a service key, traces stopped until the variable was set.
    workspace_id = (os.getenv("LANGSMITH_WORKSPACE_ID") or "").strip()
    headers = {"x-api-key": key, **({"X-Tenant-Id": workspace_id} if workspace_id else {})}
    r = client.get(f"{endpoint}/api/v1/sessions", headers=headers, params={"limit": 1})
    if r.status_code == 200:
        say("PASS", "langsmith", "can read and write projects" + (" (with LANGSMITH_WORKSPACE_ID)" if workspace_id else ""))
    elif r.status_code == 403 and not workspace_id and workspaces and len(workspaces) == 1:
        # Fixed on 2026-11-26 with a new service key limited to the one workspace (no variable needed);
        # setting the id worked locally but not on Render, for a reason never found
        say("FAIL", "langsmith", "projects refused (403): traces won't arrive. Create a service key limited to this "
                                 f"workspace, or set LANGSMITH_WORKSPACE_ID={workspaces[0].get('id')}")
    else:
        say("FAIL", "langsmith", f"projects refused (HTTP {r.status_code}): traces won't arrive. Check the key's workspace role")
    return True


# --- Redis ---
# What the app sends (services/cache_service.py, usage.py): GET SET MGET DEL EXISTS EXPIRE INCR INCRBY
# HINCRBY HGETALL, MULTI/EXEC (pipelines), PING; SCAN only in scripts/purge_url.py (the operator).
ADMIN_PROBES = (("CONFIG", ("CONFIG", "GET", "maxmemory")), ("CLIENT LIST", ("CLIENT", "LIST")))


def check_redis(_client) -> bool:
    url = (os.getenv("REDIS_URL") or "").strip()
    if not url:
        return False
    scheme = urlsplit(url).scheme
    if scheme == "rediss":
        say("PASS", "redis", "TLS (rediss://): the data and the password are encrypted on the way")
    else:
        say("FAIL", "redis", f"{scheme}:// sends the password and the data in plain text: use rediss://")
    import redis                                   # only needed here
    try:
        r = redis.from_url(url, socket_connect_timeout=5, socket_timeout=5)
        user = r.execute_command("ACL", "WHOAMI")
        user = user.decode() if isinstance(user, bytes) else str(user)
        say("WARN" if user == "default" else "PASS", "redis",
            f"logs in as {user!r}" + (": the default user usually may run every command" if user == "default" else ""))
    except redis.exceptions.NoPermissionError:
        say("PASS", "redis", "logs in as a limited user (it may not even run ACL WHOAMI)")
    except redis.exceptions.ResponseError as e:
        say("INFO", "redis", f"ACL WHOAMI isn't available ({str(e)[:80]})")
    except redis.exceptions.RedisError as e:
        say("FAIL", "redis", f"couldn't connect: {type(e).__name__}")
        return True
    for name, command in ADMIN_PROBES:
        try:
            r.execute_command(*command)
            say("WARN", "redis", f"{name} is allowed: the app's user may run admin commands it never needs")
        except redis.exceptions.NoPermissionError:
            say("PASS", "redis", f"{name} refused (NOPERM)")
        except redis.exceptions.ResponseError as e:
            say("PASS", "redis", f"{name} refused by the provider ({str(e)[:60]})")
    return True


CHECKS = {"gemini": check_gemini, "cloudflare": check_cloudflare, "langsmith": check_langsmith, "redis": check_redis}
NAMES = {"gemini": "GOOGLE_API_KEY", "cloudflare": "CLOUDFLARE_API_TOKEN + CLOUDFLARE_ACCOUNT_ID",
         "langsmith": "LANGSMITH_API_KEY", "redis": "REDIS_URL"}


def main():
    wanted = sys.argv[1:] or list(CHECKS)
    unknown = [name for name in wanted if name not in CHECKS]
    if unknown:
        print(__doc__)
        sys.exit(2)
    ran = 0
    with httpx.Client(timeout=20) as client:
        for name in wanted:
            try:
                if CHECKS[name](client):
                    ran += 1
                else:
                    say("SKIP", name, f"{NAMES[name]} not set")
            except httpx.HTTPError as e:
                say("FAIL", name, f"couldn't reach the service: {type(e).__name__}")
                ran += 1
    print("\nGNews and Cohere keys have no scopes to set: their limits are the plan's quotas (Section L counts GNews).")
    counts = {s: sum(1 for status, _, _ in results if status == s) for s in ("PASS", "WARN", "FAIL")}
    print(f"{counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail")
    sys.exit(2 if not ran else 1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()
