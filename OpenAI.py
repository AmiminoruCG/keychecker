"""OpenAI key validation using a bounded, real inference request.

A model listing proves authentication, not usable billing. Rate-limit headers
belong to the probed model/project; a matching published tier is only an estimate.
"""
import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import random
import re

import aiohttp
from APIKey import APIKey


oai_api_url = "https://api.openai.com/v1"
MAX_MODEL_PROBES = 3
MAX_RETRY_DELAY = 10
MAX_RETRY_WAIT = 20
unknown = "Tier not exposed by API"

# Published short-context limits checked 2026-09-26. Match BOTH RPM and TPM.
# https://developers.openai.com/api/docs/models/<model>
SMALL_MODEL_TIERS = (
    (500, 200_000), (5_000, 2_000_000), (5_000, 4_000_000),
    (10_000, 10_000_000), (30_000, 150_000_000),
)
MODEL_TIERS = {
    "gpt-4o-mini": SMALL_MODEL_TIERS,
    "gpt-4.1-nano": SMALL_MODEL_TIERS,
    "gpt-4o": (
        (500, 30_000), (5_000, 450_000), (5_000, 800_000),
        (10_000, 2_000_000), (10_000, 30_000_000),
    ),
    "gpt-5": (
        (500, 500_000), (5_000, 1_000_000), (5_000, 2_000_000),
        (10_000, 4_000_000), (15_000, 40_000_000),
    ),
}
MODEL_PREFERENCE = (
    "gpt-4.1-nano", "gpt-4o-mini", "gpt-5-nano", "gpt-5-mini",
    "gpt-4.1-mini", "gpt-4.1", "gpt-4o", "gpt-5",
)
RETIRED_MODELS = {
    "chatgpt-4o-latest", "gpt-5-chat-latest", "gpt-5.1-chat-latest",
    "gpt-5.2-chat-latest", "gpt-5.3-chat-latest", "gpt-5-codex",
    "gpt-5.1-codex", "gpt-5.1-codex-mini", "gpt-5.1-codex-max",
    "gpt-5.2-codex", "codex-mini-latest", "o1-preview", "o1-mini",
}
BILLING_CODES = {
    "insufficient_quota", "credit_balance_exhausted", "billing_not_active",
    "billing_hard_limit_reached", "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded", "organization_usage_limit_exceeded",
}
BLOCK_CODES = {
    "access_terminated", "account_deactivated", "organization_deactivated",
    "unsupported_country_region_territory", "ip_not_authorized",
    "ip_not_allowed", "ip_not_in_allowlist",
}


def is_openai_key(value):
    """Route scoped keys without relying on their internal marker or length."""
    return bool(
        re.fullmatch(r"sk-(?:proj|svcacct)-[A-Za-z0-9_-]{20,}", value)
        or re.fullmatch(r"sk-[A-Za-z0-9_-]+T3BlbkFJ[A-Za-z0-9_-]+", value)
    )


def model_family(model):
    if model.startswith("ft:"):
        model = model.split(":", 2)[1]
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)


def order_oai_models(models):
    candidates = []
    for model in set(models):
        family = model_family(model)
        if model in RETIRED_MODELS or family in RETIRED_MODELS:
            continue
        if not re.match(r"^(gpt-\d|o[1-9](?:-|$)|chatgpt-)", family):
            continue
        if any(part in family for part in (
            "audio", "realtime", "transcribe", "tts", "search", "image",
            "instruct", "deep-research", "-pro", "-base", "-preview",
        )):
            continue
        candidates.append(model)

    def rank(model):
        family = model_family(model)
        if family in MODEL_PREFERENCE:
            priority = MODEL_PREFERENCE.index(family)
        else:
            priority = len(MODEL_PREFERENCE) + (
                0 if any(part in family for part in ("nano", "mini", "luna")) else 1
            )
        return model.startswith("ft:"), priority, model != family, model

    return sorted(candidates, key=rank)


def safe_int(value):
    try:
        number = int(value)
        return number if number >= 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def redact_error(value, secret):
    return re.sub(r"sk-[A-Za-z0-9_.*-]+", "[REDACTED]", str(value).replace(secret, "[REDACTED]"))[:600]


def error_fields(payload):
    error = payload.get("error")
    return error if isinstance(error, dict) else {}


def billing_error(payload):
    error = error_fields(payload)
    code, kind = error.get("code"), error.get("type")
    if isinstance(code, str) and code in BILLING_CODES:
        return code
    if isinstance(kind, str) and kind in BILLING_CODES:
        return kind
    return ""


def retry_delay(headers, attempt):
    value = headers.get("retry-after")
    if value is not None:
        try:
            delay = float(value)
        except (TypeError, ValueError):
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                delay = (date - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                delay = -1
        if math.isfinite(delay) and delay >= 0:
            # Never shorten a server delay. A long delay defers this check.
            return delay if delay <= MAX_RETRY_DELAY else None
    return min(0.5 * 2 ** attempt + random.uniform(0, 0.25), MAX_RETRY_DELAY)


async def request_json(session, method, path, key, retries=3, org=None, payload=None, timeout=20):
    headers = {"Authorization": f"Bearer {key.api_key}", "Accept": "application/json"}
    if org:
        headers["OpenAI-Organization"] = org
    attempts = max(1, min(retries, 3))
    waited = 0
    for attempt in range(attempts):
        try:
            kwargs = {
                "headers": headers,
                "timeout": aiohttp.ClientTimeout(total=timeout, connect=min(timeout, 10)),
                "allow_redirects": False,
            }
            if payload is not None:
                kwargs["json"] = payload
            request = session.get if method == "GET" else session.post
            async with request(f"{oai_api_url}/{path}", **kwargs) as response:
                status = response.status
                response_headers = {k.lower(): v for k, v in response.headers.items()}
                try:
                    data = await response.json(content_type=None)
                except (TypeError, ValueError):
                    data = {}
                if not isinstance(data, dict):
                    data = {}
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            status, response_headers = 0, {}
            data = {"error": {"type": "network_error", "code": type(exc).__name__,
                              "message": "Request could not be completed; retry the check later."}}
            # A POST may have been billed before the connection was lost.
            if method != "GET":
                return status, data, response_headers
        transient = status in (0, 408, 409, 429) or 500 <= status <= 599
        if not transient or billing_error(data) or attempt + 1 == attempts:
            return status, data, response_headers
        delay = retry_delay(response_headers, attempt)
        if delay is None or waited + delay > MAX_RETRY_WAIT:
            return status, data, response_headers
        await asyncio.sleep(delay)
        waited += delay
    return status, data, response_headers


def capture_response(key, payload, headers):
    error = error_fields(payload)
    key.error_type = str(error.get("type") or "")
    key.error_code = str(error.get("code") or "")
    key.error_message = redact_error(error.get("message") or "", key.api_key)
    key.request_id = str(headers.get("x-request-id") or "")
    key.retry_after = str(headers.get("retry-after") or "")
    for header, attr in (
        ("x-ratelimit-limit-requests", "rpm"),
        ("x-ratelimit-limit-tokens", "tpm"),
        ("x-ratelimit-remaining-requests", "requests_remaining"),
        ("x-ratelimit-remaining-tokens", "tokens_remaining"),
    ):
        setattr(key, attr, safe_int(headers.get(header)))
    key.requests_reset = str(headers.get("x-ratelimit-reset-requests") or "")
    key.tokens_reset = str(headers.get("x-ratelimit-reset-tokens") or "")
    if not key.default_org and headers.get("openai-organization"):
        key.default_org = headers["openai-organization"]


def apply_error(key, status, payload):
    """None = invalid credentials; False = inconclusive; True = known billing block."""
    key.has_quota = None
    billing = billing_error(payload)
    if billing:
        key.validation_status = "valid"
        key.has_quota = False
        key.quota_status = billing
        return True
    message = key.error_message.lower()
    if (key.error_code in BLOCK_CODES or key.error_type in BLOCK_CODES
            or "ip allowlist" in message or "ip address is not" in message):
        key.blocked = True
        key.quota_status = "access_blocked"
        if key.validation_status != "valid":
            key.validation_status = "restricted"
        return False
    if status == 401:
        context_error = any(part in message for part in (
            "missing scopes", "insufficient permissions", "organization", "project",
            "not authorized", "ip address",
        )) and key.error_code != "invalid_api_key"
        if not context_error:
            key.validation_status = "invalid"
            key.quota_status = "invalid_key"
            return None
    if status in (401, 403):
        key.permission_restricted = True
        key.quota_status = "permission_restricted"
        if key.validation_status != "valid":
            key.validation_status = "restricted"
    elif status == 429:
        key.rate_limited = True
        key.quota_status = "rate_limited"
    elif status == 0:
        key.quota_status = "network_error"
    elif status >= 500:
        key.quota_status = "api_error"
    elif status == 404:
        key.quota_status = "no_usable_model"
    else:
        key.quota_status = "probe_error"
    return False


async def get_oai_model(key, session, retries=3, org=None):
    status, data, headers = await request_json(session, "GET", "models", key, retries, org)
    capture_response(key, data, headers)
    if status == 200:
        if not isinstance(data.get("data"), list):
            key.quota_status = "invalid_response"
            return False
        key.validation_status = "valid"
        key.access_to_model_listing = True
        key.models = sorted({item["id"] for item in data["data"]
                             if isinstance(item, dict) and isinstance(item.get("id"), str)})
        key.has_special_models = any(model.startswith("ft:") for model in key.models)
        candidates = order_oai_models(key.models)
        key.model = candidates[0] if candidates else ""
        return True
    result = apply_error(key, status, data)
    if status in (401, 403) and key.permission_restricted and not key.blocked:
        key.access_to_model_listing = False
        # A restricted key may allow inference while forbidding model listing.
        return True
    return result


async def get_oai_key_tier(tpm, model_name, rpm=None):
    limits = MODEL_TIERS.get(model_family(model_name))
    if limits and rpm is not None and tpm is not None:
        for tier, pair in enumerate(limits, 1):
            if pair == (rpm, tpm):
                return f"Tier {tier} (estimate)"
    return unknown


def successful_probe(path, data):
    if path == "responses":
        return (data.get("object") == "response" and data.get("status") in ("completed", "incomplete")
                and isinstance(data.get("output"), list) and not data.get("error"))
    choices = data.get("choices")
    return isinstance(choices, list) and any(
        isinstance(item, dict) and isinstance(item.get("message"), dict) for item in choices
    ) and not data.get("error")


def probe_body(model, path):
    if path == "responses":
        return {"model": model, "input": "Reply OK.", "max_output_tokens": 16, "store": False}
    return {"model": model, "messages": [{"role": "user", "content": "Reply OK."}],
            "max_completion_tokens": 1, "store": False}


async def get_oai_key_attribs(key, session, retries=3, org=None, check_id=False):
    key.has_quota = None
    key.rate_limited = False
    key.permission_restricted = False
    candidates = order_oai_models(key.models)
    if not candidates and key.access_to_model_listing is False:
        candidates = list(MODEL_PREFERENCE[:MAX_MODEL_PROBES])
    if not candidates and key.model:
        candidates = order_oai_models([key.model])
    if not candidates:
        key.quota_status = "no_suitable_model"
        return False
    for model in candidates[:MAX_MODEL_PROBES]:
        key.model = model
        path = "responses" if "codex" in model else "chat/completions"
        status, data, headers = await request_json(
            session, "POST", path, key, retries, org, probe_body(model, path)
        )
        # Some text models expose only Responses. Try it only for a specific
        # endpoint compatibility error, never for billing or transport errors.
        error = error_fields(data)
        message = str(error.get("message") or "").lower()
        if (status in (400, 404) and path == "chat/completions" and not billing_error(data)
                and ("responses" in message or "not a chat model" in message)):
            path = "responses"
            status, data, headers = await request_json(
                session, "POST", path, key, retries, org, probe_body(model, path)
            )
        capture_response(key, data, headers)
        key.tier = await get_oai_key_tier(key.tpm, model, key.rpm)
        key.tier_reason = (
            "published limits match; project/custom limits can differ"
            if key.tier != unknown else "model-specific or custom limits; account tier cannot be inferred"
        )
        if status == 200:
            if not successful_probe(path, data):
                key.quota_status = "invalid_response"
                return False
            key.validation_status = "valid"
            key.has_quota = True
            key.quota_status = "available"
            key.permission_restricted = False
            key.rate_limited = False
            key.blocked = False
            usage = data.get("usage") or {}
            if isinstance(usage, dict):
                key.input_tokens_used = safe_int(usage.get("prompt_tokens", usage.get("input_tokens"))) or 0
                key.output_tokens_used = safe_int(usage.get("completion_tokens", usage.get("output_tokens"))) or 0
                key.usage_tokens = safe_int(usage.get("total_tokens")) or key.input_tokens_used + key.output_tokens_used
            if check_id:
                key.id_verified = await check_id_verified(key, session, retries, org)
            return True
        result = apply_error(key, status, data)
        if result is None or result is True or key.blocked:
            return result
        # A missing or disallowed model does not invalidate the credential.
        if status == 404 or (status == 403 and key.error_code in ("model_not_found", "model_not_allowed")):
            continue
        return result
    return False


async def check_id_verified(key, session, retries=3, org=None):
    """Compatibility flag: o3 access is observable; identity verification isn't."""
    status, data, _ = await request_json(
        session, "POST", "chat/completions", key, retries, org, probe_body("o3", "chat/completions")
    )
    message = str(error_fields(data).get("message") or "").lower()
    if status in (400, 403) and "organization must be verified" in message:
        key.org_verification_status = "verification_required"
        return False
    if status == 200 and successful_probe("chat/completions", data):
        key.org_verification_status = "o3_access_confirmed"
    else:
        key.org_verification_status = "unknown"
    return None


async def get_oai_org(key, session, retries=1):
    """Best-effort metadata: /me is legacy and must never gate validation."""
    status, data, _ = await request_json(session, "GET", "me", key, 1, timeout=5)
    orgs = data.get("orgs")
    if status != 200 or not isinstance(orgs, dict) or not isinstance(orgs.get("data"), list):
        key.org_metadata_status = "unavailable"
        return False
    key.org_metadata_status = "available"
    for org in orgs["data"]:
        if not isinstance(org, dict) or not isinstance(org.get("id"), str) or not org["id"]:
            continue
        org_id = org["id"]
        if org_id not in key.organizations:
            key.organizations.append(org_id)
        key.organization_names[org_id] = str(org.get("name") or org_id)
        if org.get("is_default"):
            key.default_org = org_id
    return True


async def check_openai(key, session, retries=3, check_id=False, org=None, check_orgs=True):
    # Reset per-run observations; repeated checks must not retain old quota.
    fresh = APIKey(key.provider, key.api_key)
    key.__dict__.update(fresh.__dict__)
    if org:
        key.default_org = org
    result = await get_oai_model(key, session, retries, org)
    if result is None:
        return None
    if result and key.has_quota is None:
        result = await get_oai_key_attribs(key, session, retries, org, check_id)
    if check_orgs and key.validation_status == "valid":
        await get_oai_org(key, session)
    return result


async def clone_key(key, session, retries=3, check_id=False):
    """Only legacy user keys can select other organizations."""
    cloned_keys = set()
    if key.api_key.startswith(("sk-proj-", "sk-svcacct-")) or not key.default_org:
        return cloned_keys
    for org in key.organizations:
        if org == key.default_org:
            continue
        new_key = APIKey(key.provider, key.api_key)
        await check_openai(new_key, session, retries, check_id, org, check_orgs=False)
        if openai_key_usable(new_key):
            new_key.organization_names = dict(key.organization_names)
            cloned_keys.add(new_key)
    return cloned_keys


def openai_key_usable(key):
    return (key.validation_status == "valid" and key.has_quota is True
            and not key.rate_limited and not key.permission_restricted and not key.blocked)


def openai_status_text(key):
    labels = {
        "available": "usable quota (inference confirmed)",
        "insufficient_quota": "quota exhausted",
        "credit_balance_exhausted": "credit balance exhausted",
        "billing_not_active": "billing inactive",
        "billing_hard_limit_reached": "billing limit reached",
        "organization_spend_limit_exceeded": "organization spend limit reached",
        "project_spend_limit_exceeded": "project spend limit reached",
        "organization_usage_limit_exceeded": "organization usage limit reached",
        "rate_limited": "temporary rate limit; quota unknown",
        "permission_restricted": "permission restricted; quota unknown",
        "access_blocked": "access blocked; quota unknown",
        "network_error": "network error; retry later",
        "api_error": "temporary API error; retry later",
        "probe_error": "probe failed; quota unknown",
        "invalid_response": "unexpected API response; quota unknown",
        "no_usable_model": "tested models unavailable; quota unknown",
        "no_suitable_model": "no suitable text model for a small probe; quota unknown",
    }
    return labels.get(key.quota_status, key.quota_status)


def format_key_details(key):
    details = [key.api_key, openai_status_text(key)]
    if key.model:
        details.append(f"model: {key.model}")
    if key.rpm is not None:
        details.append(f"{key.rpm:,} RPM")
    if key.tpm is not None:
        details.append(f"{key.tpm:,} TPM")
    if key.has_quota is True:
        details.append(key.tier or unknown)
        if key.tier == unknown:
            details.append(key.tier_reason)
    if key.requests_remaining is not None:
        details.append(f"requests remaining: {key.requests_remaining:,}")
    if key.tokens_remaining is not None:
        details.append(f"tokens remaining: {key.tokens_remaining:,}")
    if key.retry_after:
        details.append(f"Retry-After: {key.retry_after}")
    if key.default_org:
        details.append(f"org: {key.default_org}")
    if key.access_to_model_listing is False:
        details.append("model listing restricted")
    if key.org_metadata_status == "unavailable":
        details.append("organization metadata unavailable")
    if key.has_special_models:
        details.append("fine-tuned models listed")
    if key.org_verification_status != "not_checked":
        details.append({
            "o3_access_confirmed": "o3 access confirmed; identity verification not exposed by API",
            "verification_required": "organization verification required",
            "unknown": "organization verification unknown",
        }[key.org_verification_status])
    if key.usage_tokens:
        details.append(f"probe usage: {key.input_tokens_used} input + {key.output_tokens_used} output tokens")
    if key.has_quota is not True and key.error_code:
        details.append(f"error: {key.error_code}")
    if key.has_quota is None and key.error_message:
        details.append(key.error_message.replace("\n", " "))
    return " | ".join(details)


def pretty_print_oai_keys(keys, cloned_keys=()):
    print("-" * 90)
    groups = {
        "Usable quota (inference confirmed)": [],
        "Valid keys with billing/quota exhausted": [],
        "Authenticated keys; quota not confirmed": [],
        "Unconfirmed or restricted keys; retry/check access": [],
    }
    for key in keys:
        if openai_key_usable(key):
            group = "Usable quota (inference confirmed)"
        elif key.validation_status == "valid" and key.has_quota is False:
            group = "Valid keys with billing/quota exhausted"
        elif key.validation_status == "valid":
            group = "Authenticated keys; quota not confirmed"
        else:
            group = "Unconfirmed or restricted keys; retry/check access"
        groups[group].append(key)
    for label, items in groups.items():
        if items:
            print(f"\n--- OpenAI: {label}: {len(items)} ---")
            for key in sorted(items, key=lambda k: (k.api_key, k.default_org)):
                print(format_key_details(key))
    confirmed = sum(key.validation_status == "valid" for key in keys)
    usable = sum(openai_key_usable(key) for key in keys)
    print(f"\n--- OpenAI: {confirmed} authenticated, {usable} with usable quota, "
          f"{len(keys) - confirmed} unconfirmed/restricted ---")
    if any(key.has_quota is True for key in keys):
        print("Tier estimates compare this model's RPM/TPM with published defaults; "
              "the account tier is not exposed by these endpoints.")
    if cloned_keys:
        print(f"Additional organization contexts with confirmed inference: {len(cloned_keys)}")
