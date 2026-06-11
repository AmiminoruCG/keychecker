import random
import APIKey

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
ALIVE_CHECK_MODEL = "gemini-2.5-pro"
TTS_TIER_MODEL = "gemini-2.5-pro-preview-tts"
IMAGEN_BILLING_MODEL = "imagen-4.0-generate-001"

tracked_models = {
    "gemini-3.5-flash": "has 3.5 flash",
    "gemini-2.5-pro": "has 2.5 pro",
    "gemini-3.1-flash-tts-preview": "has 3.1 tts",
    "gemini-2.5-pro-preview-tts": "has 2.5 pro tts",
    "imagen-4.0-generate-001": "has imagen 4",
}


def gemini_headers(key: APIKey):
    return {"x-goog-api-key": key.api_key}


async def response_json(response):
    try:
        return await response.json()
    except Exception:
        return {}


def normalize_model_name(model_name):
    return model_name.replace("models/", "").replace("-latest", "")


async def check_makersuite(key: APIKey, session):
    async with session.get(f"{GEMINI_API_BASE}/models", headers=gemini_headers(key)) as response:
        resp_json = await response_json(response)
        models = resp_json.get("models", [])
        if response.status != 200 or not await test_key_alive(key, session):
            return
        key.enabled_billing = await test_makersuite_billing(key, session)
        model_names = {normalize_model_name(model.get("name", "")) for model in models}
        key.models.extend(model for model in tracked_models if model in model_names)
        if key.enabled_billing:
            await test_key_tier(key, session)
        else:
            key.tier = "Free Tier"
        return True


async def test_key_alive(key: APIKey, session):
    data = {"generationConfig": {"max_output_tokens": 0}}
    async with session.post(f"{GEMINI_API_BASE}/models/{ALIVE_CHECK_MODEL}:generateContent", headers=gemini_headers(key), json=data) as response:
        resp_json = await response_json(response)
        if response.status == 429:
            error_details = resp_json.get("error", {}).get("message", "")
            # different type of 429 error compared to hitting the rpm limit, keys with this seem to never recover and are just perma 429'd, so we mark them as invalid
            if "limit 'GenerateContent request limit per minute for a region' of service 'generativelanguage.googleapis.com' for consumer" in error_details:
                return False
            if any(violation.get("quotaValue") == "0" for violation in quota_violations(resp_json)):
                return False
        return response.status in (200, 400, 429)


def quota_violations(resp_json):
    violations = []
    for detail in resp_json.get("error", {}).get("details", []):
        violations.extend(detail.get("violations", []))
    return violations


def infer_tier_from_violations(violations):
    for violation in violations:
        quota_text = " ".join(str(violation.get(field, "")).lower() for field in ("quotaMetric", "quotaId"))
        if "tier_3" in quota_text or "tier3" in quota_text or "-tier3" in quota_text:
            return "Tier 3"
        if "tier_2" in quota_text or "tier2" in quota_text or "-tier2" in quota_text:
            return "Tier 2"
        if "tier_1" in quota_text or "tier1" in quota_text or "paid_tier" in quota_text:
            return "Tier 1"
    return ""


def format_quota_details(violations):
    if not violations:
        return "Unknown Tier"
    violation = violations[0]
    return f"(QM {violation.get('quotaMetric', '')} | QV {violation.get('quotaValue', '')})"


async def test_key_tier(key: APIKey, session):
    data = {
        "contents": [{
            "parts": [{
                "text": "hello" * random.randint(66666, 77777),
            }]
        }],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": "Kore",
                    },
                },
            },
        },
        "model": TTS_TIER_MODEL,
    }

    async with session.post(f"{GEMINI_API_BASE}/models/{TTS_TIER_MODEL}:generateContent", headers=gemini_headers(key), json=data) as response:
        resp_json = await response_json(response)
        if response.status == 200:
            key.tier = "??? Tier"
        elif response.status == 400:
            if "exceeds the maximum number of tokens allowed" in resp_json.get("error", {}).get("message", ""):
                key.tier = "Tier 3"
            else:
                key.tier = "Unknown Tier"
        elif response.status == 429:
            violations = quota_violations(resp_json)
            key.tier = infer_tier_from_violations(violations) or format_quota_details(violations)
        else:
            key.tier = "Unknown Tier"


async def test_makersuite_billing(key: APIKey, session):
    data = {"instances": [{"prompt": ""}]}
    async with session.post(f"{GEMINI_API_BASE}/models/{IMAGEN_BILLING_MODEL}:predict", headers=gemini_headers(key), json=data) as response:
        resp_json = await response_json(response)
        if response.status in (200, 429):
            return True
        if response.status == 400:
            error_details = resp_json.get("error", {}).get("message", "")
            if "Imagen API is only accessible to billed users at this time" not in error_details:
                return True
        return False


def pretty_print_makersuite_keys(keys):
    total = 0
    billing_count = 0
    model_counts = {model: 0 for model in tracked_models}

    print('-' * 90)
    print(f'Validated {len(keys)} MakerSuite keys:')

    keys_by_tier = {}
    unknown_keys = set()
    output_order = [
        "Free Tier",
        "Tier 1",
        "Tier 2",
        "Tier 3",
        "??? Tier",
        "Unknown Tier",
    ]

    for key in keys:
        total += 1
        if key.enabled_billing:
            billing_count += 1
        if "(" in key.tier:
            unknown_keys.add(key)
        else:
            keys_by_tier.setdefault(key.tier, []).append(key)

    for tier in output_order:
        if tier in keys_by_tier:
            keys_in_tier = keys_by_tier[tier]
            print(f'\n{len(keys_in_tier)} keys found in {tier}:')
            for key in keys_in_tier:
                model_labels = [label for model, label in tracked_models.items() if model in key.models]
                print(f'{key.api_key}' + (f" | {', '.join(model_labels)}" if model_labels else ''))
                for model in key.models:
                    if model in model_counts:
                        model_counts[model] += 1

    if len(unknown_keys) > 0:
        print(f"Found {len(unknown_keys)} keys with strange quota values")
        for key in unknown_keys:
            model_labels = [label for model, label in tracked_models.items() if model in key.models]
            print(key.api_key + " | " + key.tier + (f" | {', '.join(model_labels)}" if model_labels else ''))
            for model in key.models:
                if model in model_counts:
                    model_counts[model] += 1

    model_summary = ', '.join(f"{count} {tracked_models[model]}" for model, count in model_counts.items() if count > 0)
    print(f'\n--- Total Valid MakerSuite Keys: {total} ({billing_count} with billing enabled'
          + (f', {model_summary}) ---\n' if model_summary else ') ---\n'))
