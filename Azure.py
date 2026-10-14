"""Azure resource-key validation: authentication and inference are separate checks."""

import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import unquote, urljoin, urlsplit

import requests

from APIKey import APIKey, Provider


TIMEOUT = (5, 20)
MAX_OUTPUT_TOKENS = 16
MAX_PROBES = 20
MAX_PAGES = 10
CHECK_SECONDS = 90
MAX_RETRY_DELAY = 10
AZURE_SUFFIXES = (
    "openai.azure.com", "services.ai.azure.com", "cognitiveservices.azure.com",
    "openai.azure.us", "cognitiveservices.azure.us",
    "openai.azure.cn", "cognitiveservices.azure.cn",
)
RESOURCE_RE = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,62}[A-Za-z0-9])?\Z")
SECRET_RE = re.compile(r"[A-Za-z0-9_-]{20,256}\Z")
DEPLOYMENT_RE = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
BILLING_CODES = {
    "insufficient_quota", "credit_balance_exhausted", "billing_not_active",
    "billing_hard_limit_reached", "usage_limit_reached", "spend_limit_reached",
}


def parse_azure_key(value):
    """Return an allowlisted HTTPS origin, secret, and optional deployment ID."""
    if not isinstance(value, str) or ":" not in value.strip():
        raise ValueError("Expected resource-name:apikey or HTTPS-endpoint:apikey")
    if re.fullmatch(r"(?:AKIA|ASIA)[A-Z0-9]{16}:[A-Za-z0-9/+=]{40}", value.strip()):
        raise ValueError("AWS credentials are not Azure resource keys")
    address, secret = value.strip().rsplit(":", 1)
    if not SECRET_RE.fullmatch(secret):
        raise ValueError("Invalid Azure key format")
    if "://" not in address:
        if RESOURCE_RE.fullmatch(address):
            address += ".openai.azure.com"
        address = "https://" + address
    try:
        parsed = urlsplit(address)
        host = parsed.hostname or ""
        port = parsed.port
    except ValueError as error:
        raise ValueError("Invalid Azure endpoint") from error
    if (parsed.scheme != "https" or parsed.username or parsed.password or port is not None
            or parsed.fragment or not any(
                host.endswith("." + suffix)
                and RESOURCE_RE.fullmatch(host[:-(len(suffix) + 1)])
                for suffix in AZURE_SUFFIXES)):
        raise ValueError("Expected an HTTPS Azure resource endpoint")
    path = parsed.path.rstrip("/")
    hint = ""
    match = re.fullmatch(r"/openai/deployments/([^/]+)(?:/(?:chat/completions|responses))?", path)
    if match:
        hint = unquote(match[1])
        if not DEPLOYMENT_RE.fullmatch(hint):
            raise ValueError("Invalid deployment name")
    elif path not in ("", "/openai", "/openai/v1", "/openai/v1/chat/completions", "/openai/v1/responses"):
        raise ValueError("Unsupported Azure endpoint path")
    return "https://" + host.lower(), secret, hint


def is_azure_key(value):
    try:
        parse_azure_key(value)
        return True
    except ValueError:
        return False


def _text(value, secret=""):
    text = str(value) if value is not None else ""
    if secret:
        text = text.replace(secret, "[redacted]")
    return " ".join(text.split())[:400]


def _error(payload, secret=""):
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    if not isinstance(error, dict):
        return "invalid_response", _text(error, secret)
    return _text(error.get("code") or error.get("type"), secret), _text(error.get("message"), secret)


def _number(value):
    try:
        result = int(value)
        return result if result >= 0 and not isinstance(value, bool) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _retry_delay(headers, attempt):
    delays = []
    for name in ("retry-after", "retry-after-ms", "x-ms-retry-after-ms"):
        value = headers.get(name)
        if value is None:
            continue
        try:
            delay = float(value)
            if name.endswith("-ms"):
                delay /= 1000
        except (TypeError, ValueError):
            try:
                stamp = parsedate_to_datetime(str(value))
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                delay = (stamp - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                continue
        if delay >= 0:
            delays.append(delay)
    return max(delays) if delays else min(2 ** attempt, MAX_RETRY_DELAY)


def _request(key, session, method, url, secret, retries, deadline, body=None):
    """No redirects; POST transport errors are not replayed (billing is uncertain)."""
    result = (0, {"error": {"code": "deadline_exceeded", "message": "Check time budget exhausted."}}, {})
    for attempt in range(max(1, min(retries, 3))):
        if time.monotonic() >= deadline:
            break
        try:
            key.requests_made += 1
            kwargs = {"headers": {"api-key": secret, "accept": "application/json"},
                      "timeout": TIMEOUT, "allow_redirects": False}
            if body is not None:
                kwargs["json"] = body
            response = session.request(method, url, **kwargs)
            headers = {str(name).lower(): str(value) for name, value in response.headers.items()}
            status = response.status_code
            try:
                payload = response.json()
            except (ValueError, TypeError):
                payload = {"error": {"code": "invalid_json", "message": "API returned a non-JSON response."}}
            result = status, payload, headers
        except requests.RequestException as error:
            result = 0, {"error": {"code": type(error).__name__, "message": "Network request failed; result is inconclusive."}}, {}
            if method == "POST":
                break
            status, payload, headers = result
        code, _ = _error(payload)
        if code.lower() in BILLING_CODES or status not in (0, 408, 409, 429, 500, 502, 503, 504):
            break
        delay = _retry_delay(headers, attempt)
        if attempt + 1 >= max(1, min(retries, 3)) or delay > MAX_RETRY_DELAY or time.monotonic() + delay >= deadline:
            break
        time.sleep(delay)
    return result


def _response_details(result, secret):
    status, payload, headers = result
    code, message = _error(payload, secret)
    return {"http_status": status, "error_code": code, "error_message": message,
            "request_id": _text(headers.get("apim-request-id") or headers.get("x-request-id") or headers.get("x-ms-request-id")),
            "retry_after": _text(headers.get("retry-after") or headers.get("retry-after-ms") or headers.get("x-ms-retry-after-ms"))}


def _failure_status(result):
    status, payload, _ = result
    code, message = _error(payload)
    normalized = code.lower()
    if status == 401:
        return "invalid"
    if normalized in BILLING_CODES:
        return "quota_exhausted"
    if status == 403 or normalized in ("accountdisabled", "subscriptiondisabled", "accessdenied"):
        return "restricted"
    if status == 429:
        return "rate_limited"
    if status == 404 or normalized in ("deploymentnotfound", "model_not_found"):
        return "not_found"
    if status == 400 and ("not supported" in message.lower() or "unsupported" in normalized):
        return "unsupported"
    return "unknown"


def _set_error(key, result, secret):
    for name, value in _response_details(result, secret).items():
        if name != "http_status":
            setattr(key, name, value)


def _discover(key, session, secret, retries, deadline):
    # This resource-key endpoint still works on some resources. v1/models is a
    # model catalogue, NOT a list of user deployment IDs; it cannot replace it.
    url = key.endpoint + "/openai/deployments?api-version=2022-12-01"
    seen_urls, seen_ids, deployments = set(), set(), []
    malformed = False
    for _ in range(MAX_PAGES):
        if url in seen_urls:
            key.discovery_status = "partial"
            break
        seen_urls.add(url)
        result = _request(key, session, "GET", url, secret, retries, deadline)
        status, payload, _ = result
        if status != 200:
            _set_error(key, result, secret)
            if status == 401:
                if key.validation_status != "valid":
                    key.validation_status = key.access_status = "invalid"
                    return deployments
                key.discovery_status = "partial"
                key.access_status = "restricted"
                break
            key.discovery_status = "partial" if deployments else "unavailable"
            key.access_status = _failure_status(result)
            break
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            key.discovery_status = "partial" if deployments else "invalid_response"
            key.error_code = "invalid_deployment_response"
            key.error_message = "Deployment listing returned an unexpected response schema."
            break
        key.validation_status = "valid"
        for item in payload["data"]:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not DEPLOYMENT_RE.fullmatch(item["id"]):
                malformed = True
                continue
            if item["id"] in seen_ids:
                continue
            seen_ids.add(item["id"])
            model = item.get("model", "")
            if isinstance(model, dict):
                model = model.get("name", "")
            deployments.append({"id": item["id"], "model": _text(model),
                                "provisioning_status": _text(item.get("status", "unknown")).lower(),
                                "status": "not_checked"})
        next_link = payload.get("nextLink") or payload.get("next_link")
        if not next_link:
            key.discovery_status = "partial" if malformed else "complete"
            key.discovery_complete = not malformed
            return deployments
        if not isinstance(next_link, str):
            key.discovery_status = "partial"
            break
        candidate = urljoin(url, next_link)
        parsed = urlsplit(candidate)
        if (parsed.scheme + "://" + parsed.netloc != key.endpoint
                or parsed.path.rstrip("/") != "/openai/deployments" or parsed.fragment):
            key.discovery_status = "partial"
            key.error_code = "unsafe_pagination_link"
            key.error_message = "Deployment pagination did not stay on the resource endpoint."
            break
        url = candidate
    else:
        key.discovery_status = "partial"
    return deployments


def _supported_text(model):
    name = model.lower()
    if any(part in name for part in ("audio", "realtime", "transcribe", "tts", "embedding", "image", "dall-e", "whisper", "search", "instruct")):
        return False
    return bool(re.match(r"^(?:gpt-(?:[4-9]|35)|o[1-9])(?:[.\w-]*)$", name))


def _model_rank(deployment):
    # Prefer newer verified families, then the full model within that family.
    model = deployment["model"].lower()
    match = re.match(r"gpt-(\d+)(?:\.(\d+))?", model)
    if match:
        major = int(match[1])
        if major == 35:
            major, minor = 3, 5
        else:
            minor = int(match[2] or 0)
        revision = 2 if model.startswith("gpt-4o") else 1 if "turbo" in model else 0
        family = (major, minor, revision)
    else:
        match = re.match(r"o(\d+)", model)
        family = (4, int(match[1]) if match else -1, 0)
    size = -2 if "nano" in model else -1 if "mini" in model else 0
    return (*family, size, deployment["id"])


def _probe(key, deployment, session, secret, retries, deadline):
    model = deployment["model"].lower()
    api = "responses" if "codex" in model or re.search(r"(?:^|-)pro(?:-|$)", model) else "chat/completions"
    body = {"model": deployment["id"], "store": False}
    if api == "responses":
        body.update(input="Reply OK.", max_output_tokens=MAX_OUTPUT_TOKENS)
    else:
        body.update(messages=[{"role": "user", "content": "Reply OK."}], max_completion_tokens=MAX_OUTPUT_TOKENS)
        if re.match(r"gpt-5\.[1-9](?:-|$)", model):
            body["reasoning_effort"] = "none"
    result = _request(key, session, "POST", key.endpoint + "/openai/v1/" + api,
                      secret, retries, deadline, body)
    # Unknown explicit deployments can tell us that only Responses is supported.
    code, message = _error(result[1])
    if (api == "chat/completions" and result[0] == 400
            and "responses" in message.lower() and ("not supported" in message.lower() or "unsupported" in code.lower())):
        api = "responses"
        result = _request(key, session, "POST", key.endpoint + "/openai/v1/responses", secret, retries, deadline,
                          {"model": deployment["id"], "input": "Reply OK.", "max_output_tokens": MAX_OUTPUT_TOKENS, "store": False})
    status, payload, headers = result
    deployment.update(_response_details(result, secret), api=api, status=_failure_status(result))
    if status == 401 and key.validation_status == "valid":
        # Discovery already authenticated this key; a different endpoint's
        # denial must not erase that evidence or hide the credential.
        deployment["status"] = "restricted"
    if status != 200:
        return
    if not isinstance(payload, dict) or payload.get("error"):
        deployment.update(status="unknown", error_code=deployment["error_code"] or "invalid_response")
        return
    usage = payload.get("usage", {})
    usage_valid = False
    input_tokens = output_tokens = 0
    if isinstance(usage, dict):
        input_value = usage.get("prompt_tokens", usage.get("input_tokens"))
        output_value = usage.get("completion_tokens", usage.get("output_tokens"))
        usage_valid = (type(input_value) is int and input_value > 0
                       and type(output_value) is int and output_value >= 0)
        input_tokens = _number(input_value) or 0
        output_tokens = _number(output_value) or 0
        total = _number(usage.get("total_tokens"))
        deployment["usage_tokens"] = total if total is not None else input_tokens + output_tokens
        key.input_tokens_used += input_tokens
        key.output_tokens_used += output_tokens
        key.usage_tokens += deployment["usage_tokens"]
    choices = payload.get("choices")
    if api == "responses":
        incomplete = payload.get("incomplete_details") or {}
        output = payload.get("output")
        completed_message = isinstance(output, list) and any(
            isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "assistant"
            and isinstance(item.get("content"), list) and bool(item["content"]) for item in output)
        valid = (payload.get("object") == "response" and isinstance(output, list)
                 and ((payload.get("status") == "completed" and completed_message)
                 or (payload.get("status") == "incomplete" and output_tokens > 0
                     and isinstance(incomplete, dict) and incomplete.get("reason") == "max_output_tokens")))
    else:
        valid = (payload.get("object") == "chat.completion"
                 and isinstance(choices, list) and bool(choices) and isinstance(choices[0], dict)
                 and isinstance(choices[0].get("message"), dict)
                 and choices[0]["message"].get("role") == "assistant"
                 and choices[0].get("finish_reason") in ("stop", "length", "tool_calls", "function_call", "content_filter"))
        if valid and choices[0].get("finish_reason") == "content_filter":
            deployment.update(status="restricted", error_code="content_filter", error_message="Probe output was blocked by content filtering.")
            return
    if not valid or not usage_valid:
        deployment.update(status="unknown", error_code="invalid_response", error_message="Inference returned an unexpected response schema.")
        return
    key.validation_status = "valid"
    deployment.update(status="usable", error_code="", error_message="",
                      filter_status="annotations_present" if payload.get("prompt_filter_results") or
                      (isinstance(choices, list) and any(isinstance(choice, dict) and "content_filter_results" in choice for choice in choices)) else "annotations_absent",
                      rpm=_number(headers.get("x-ratelimit-limit-requests")),
                      tpm=_number(headers.get("x-ratelimit-limit-tokens")))
    # Deployment metadata is preferable to a snapshot model name in the response.
    if not deployment["model"] and isinstance(payload.get("model"), str):
        deployment["model"] = _text(payload["model"])


def check_azure(key, session=None, retries=3):
    """True: usable text inference; None: rejected key; False: inconclusive/restricted."""
    fresh = APIKey(Provider.AZURE, key.api_key)
    key.__dict__.update(fresh.__dict__)
    try:
        key.endpoint, secret, key.deployment_hint = parse_azure_key(key.api_key)
    except ValueError as error:
        key.error_code, key.error_message = "invalid_input", str(error)
        return False
    key.resource_name = urlsplit(key.endpoint).hostname.split(".")[0]
    deadline = time.monotonic() + CHECK_SECONDS
    owned_session = session is None
    if owned_session:
        session = requests.Session()
    try:
        deployments = _discover(key, session, secret, retries, deadline)
        key.deployments = deployments
        if key.validation_status == "invalid":
            return None
        if key.deployment_hint:
            matching = next((item for item in deployments if item["id"] == key.deployment_hint), None)
            if matching is None:
                matching = {"id": key.deployment_hint, "model": "", "provisioning_status": "unknown", "status": "not_checked"}
                deployments.append(matching)
            candidates = [matching]
        else:
            candidates = deployments
        probes = []
        for item in deployments:
            if item["provisioning_status"] not in ("succeeded", "ready", "unknown", ""):
                item["status"] = "failed" if item["provisioning_status"] in ("failed", "deleted", "canceled", "cancelled") else "not_ready"
            elif not _supported_text(item["model"]) and not (item["id"] == key.deployment_hint and not item["model"]):
                item["status"] = "unsupported"
            elif item in candidates:
                probes.append(item)
        # Cheap variants first; all eligible discoveries are bounded by MAX_PROBES.
        probes.sort(key=lambda item: (not any(size in item["model"] for size in ("mini", "nano")), _model_rank(item)))
        for item in probes[:MAX_PROBES]:
            if time.monotonic() >= deadline:
                break
            _probe(key, item, session, secret, retries, deadline)
            if item["status"] == "invalid":
                key.validation_status = key.access_status = "invalid"
                for name in ("error_code", "error_message", "request_id", "retry_after"):
                    setattr(key, name, item.get(name, ""))
                return None
        if key.validation_status != "valid" and not probes:
            # Authentication fallback only: catalogue IDs must never be inferred as deployments.
            result = _request(key, session, "GET", key.endpoint + "/openai/v1/models", secret, retries, deadline)
            if result[0] == 401:
                key.validation_status = key.access_status = "invalid"
                _set_error(key, result, secret)
                return None
            if (result[0] == 200 and isinstance(result[1], dict)
                    and isinstance(result[1].get("data"), list)):
                key.validation_status = "valid"
            else:
                key.access_status = _failure_status(result)
                _set_error(key, result, secret)
        usable = [item for item in deployments if item["status"] == "usable"]
        if usable:
            best = max(usable, key=_model_rank)
            key.validation_status, key.access_status, key.has_quota = "valid", "usable", True
            key.best_deployment, key.model = best["id"], best["model"]
            key.filter_status, key.rpm, key.tpm = best["filter_status"], best["rpm"], best["tpm"]
            key.request_id = best.get("request_id", "")
            key.error_code = key.error_message = key.retry_after = ""
            return True
        attempted = [item for item in probes if item["status"] != "not_checked"]
        if attempted:
            priority = {"restricted": 0, "quota_exhausted": 1, "rate_limited": 2}
            failure = min(attempted, key=lambda item: priority.get(item["status"], 3))
            key.access_status = failure["status"]
            if len(attempted) == len(probes) and all(item["status"] == "quota_exhausted" for item in attempted):
                key.has_quota = False
            for name in ("error_code", "error_message", "request_id", "retry_after"):
                setattr(key, name, failure.get(name, ""))
        elif probes:
            key.access_status = "not_checked"
            key.error_code = "probe_budget_exhausted"
            key.error_message = "Eligible deployments remain untested because the check budget was exhausted."
        elif key.validation_status == "valid":
            key.access_status = "no_usable_deployment" if key.discovery_complete else "deployment_required"
        if key.access_status == "restricted" and key.validation_status != "valid":
            key.validation_status = "restricted"
        return False
    finally:
        if owned_session:
            session.close()


def azure_key_usable(key):
    return (key.validation_status == "valid" and key.access_status == "usable" and key.has_quota is True
            and bool(key.best_deployment) and any(item.get("id") == key.best_deployment and item.get("status") == "usable"
                                                for item in key.deployments))


def azure_proxy_credential(key):
    if not azure_key_usable(key):
        return ""
    try:
        endpoint, secret, _ = parse_azure_key(key.api_key)
    except ValueError:
        return ""
    host = urlsplit(endpoint).hostname
    chosen = next(item for item in key.deployments if item.get("id") == key.best_deployment)
    if not host.endswith(".openai.azure.com") or chosen.get("api") != "chat/completions":
        return ""
    return f"{host.split('.')[0]}:{key.best_deployment}:{secret}"


def azure_status_text(key):
    if azure_key_usable(key):
        return "usable inference confirmed"
    if key.validation_status == "invalid":
        return "authentication rejected"
    return f"authentication {key.validation_status}; inference {key.access_status}"


def pretty_print_azure_keys(keys):
    print("-" * 90)
    usable_count = sum(azure_key_usable(key) for key in keys)
    print(f"Checked {len(keys)} Azure credentials ({usable_count} usable, {len(keys) - usable_count} restricted/unconfirmed):")
    for key in sorted(keys, key=lambda item: (not azure_key_usable(item), item.resource_name)):
        print(f"{key.api_key} | {azure_status_text(key)} | discovery {key.discovery_status}")
        if azure_key_usable(key):
            print(f"  preferred verified deployment: {key.best_deployment} | model: {key.model} | RPM: {key.rpm if key.rpm is not None else 'unknown'} | TPM: {key.tpm if key.tpm is not None else 'unknown'}")
            print(f"  filter annotations: {key.filter_status}; filter configuration unknown")
            if not azure_proxy_credential(key):
                print("  legacy proxy export unavailable for this endpoint/API")
        elif key.error_code or key.error_message:
            print(f"  {key.error_code}: {key.error_message}")
        if key.access_status == "deployment_required":
            print("  Supply an HTTPS deployment URL to verify inference when deployment listing is unavailable.")
        for item in key.deployments:
            detail = f" | HTTP {item['http_status']}" if item.get("http_status") else ""
            if item.get("error_code") or item.get("error_message"):
                detail += f" | {item.get('error_code', '')}: {item.get('error_message', '')}"
            print(f"  {item['id']} ({item['model'] or 'model unknown'}): {item['status']} | provisioning {item['provisioning_status']}{detail}")
        print(f"  reported probe usage: {key.usage_tokens} tokens | requests: {key.requests_made}")
    print(f"\n--- Usable Azure credentials: {usable_count}; restricted/unconfirmed: {len(keys) - usable_count} ---\n")
