import APIKey


XAI_API_BASE = 'https://api.x.ai/v1'

RATE_LIMIT_HEADERS = {
    'x-ratelimit-limit-requests': 'requests_limit',
    'x-ratelimit-remaining-requests': 'requests_remaining',
    'x-ratelimit-reset-requests': 'requests_reset',
    'x-ratelimit-limit-tokens': 'tokens_limit',
    'x-ratelimit-remaining-tokens': 'tokens_remaining',
    'x-ratelimit-reset-tokens': 'tokens_reset',
}


async def check_xai(key: APIKey, session):
    headers = {'Authorization': f'Bearer {key.api_key}'}

    async with session.get(f'{XAI_API_BASE}/api-key', headers=headers) as response:
        payload = await response_json(response)
        capture_xai_response(key, response, payload)

        if is_invalid_xai_auth(response.status, key.error_type, key.error_message):
            key.validation_status = 'invalid'
            return None
        if response.status == 403:
            key.validation_status = 'valid'
            key.permission_restricted = True
            key.access_status = 'permission_restricted'
            return True
        if response.status == 429:
            key.validation_status = 'valid'
            key.rate_limited = True
            key.access_status = 'rate_limited'
            return True
        if response.status >= 500:
            key.validation_status = 'unavailable'
            key.access_status = 'api_error'
            return False
        if response.status != 200:
            key.validation_status = 'unavailable'
            key.access_status = 'validation_error'
            return False

        key.validation_status = 'valid'
        key.team_blocked = bool(payload.get('team_blocked', False))
        key.api_key_blocked = bool(payload.get('api_key_blocked', False))
        key.api_key_disabled = bool(payload.get('api_key_disabled', False))
        key.blocked = (
            key.team_blocked
            or key.api_key_blocked
            or key.api_key_disabled
        )
        key.acls = [str(acl) for acl in (payload.get('acls') or [])]
        key.full_endpoint_access = 'api-key:endpoint:*' in key.acls
        key.full_model_access = 'api-key:model:*' in key.acls

    if key.blocked:
        key.access_status = 'blocked'
        return True

    async with session.get(f'{XAI_API_BASE}/models', headers=headers) as response:
        payload = await response_json(response)
        capture_xai_response(key, response, payload)

        if is_invalid_xai_auth(response.status, key.error_type, key.error_message):
            key.validation_status = 'invalid'
            return None
        if response.status == 403:
            key.permission_restricted = True
            key.access_status = 'permission_restricted'
            return True
        if response.status == 429:
            key.rate_limited = True
            key.access_status = 'rate_limited'
            return True
        if response.status >= 500:
            key.access_status = 'api_error'
            return True
        if response.status != 200:
            key.access_status = 'model_list_error'
            return True

        model_records = [
            item for item in (payload.get('data') or [])
            if isinstance(item, dict) and item.get('id')
        ]
        key.models = [item['id'] for item in model_records]
        key.text_models = order_xai_text_models(model_records)

    if not key.text_models:
        key.access_status = 'no_text_models'
        return True

    candidates = key.text_models[:3]
    response_headers = {**headers, 'Content-Type': 'application/json'}
    for index, model in enumerate(candidates):
        key.model = model
        data = {
            'model': model,
            'input': [{'role': 'user', 'content': 'Reply with OK.'}],
            'max_output_tokens': 1,
            'store': False,
        }

        async with session.post(
            f'{XAI_API_BASE}/responses',
            headers=response_headers,
            json=data,
        ) as response:
            payload = await response_json(response)
            capture_xai_response(key, response, payload)

            if response.status == 200:
                usage = payload.get('usage') or {}
                key.input_tokens_used = int(usage.get('input_tokens') or 0)
                key.output_tokens_used = int(usage.get('output_tokens') or 0)
                key.usage_tokens = int(
                    usage.get('total_tokens')
                    or key.input_tokens_used + key.output_tokens_used
                )
                key.cost_usd_ticks = int(usage.get('cost_in_usd_ticks') or 0)
                key.test_cost_usd = key.cost_usd_ticks / 10_000_000_000
                key.service_tier = str(payload.get('service_tier') or '')
                key.has_usable_billing = True
                key.subbed = True
                key.access_status = 'usable'
                key.error_type = ''
                key.error_message = ''
                return True

            if is_invalid_xai_auth(response.status, key.error_type, key.error_message):
                key.validation_status = 'invalid'
                return None

            access_status = classify_xai_access_error(
                response.status,
                key.error_type,
                key.error_message,
            )
            if access_status in ('no_balance', 'billing_error'):
                key.has_usable_billing = False
                key.access_status = access_status
                return True
            if access_status == 'blocked':
                key.blocked = True
                key.access_status = 'blocked'
                return True

            if response.status == 429:
                key.rate_limited = True
                key.access_status = 'rate_limited'
                return True

            if response.status in (403, 404) and index + 1 < len(candidates):
                continue
            if (
                response.status in (400, 422)
                and model_error_can_fallback(key.error_message)
                and index + 1 < len(candidates)
            ):
                continue

            if response.status == 403:
                key.permission_restricted = True
                key.access_status = 'permission_restricted'
            elif response.status == 404:
                key.access_status = 'no_usable_model'
            elif response.status >= 500:
                key.access_status = 'api_error'
            else:
                key.access_status = 'probe_error'
            return True

    key.access_status = 'no_usable_model'
    return True


async def response_json(response):
    try:
        payload = await response.json(content_type=None)
        return payload if isinstance(payload, dict) else {}
    except (TypeError, ValueError):
        return {}


def capture_xai_response(key, response, payload):
    parse_xai_rate_limits(key, response.headers)
    key.error_type, key.error_message = parse_xai_error(payload)
    key.request_id = str(
        response.headers.get('x-request-id')
        or payload.get('request_id')
        or ''
    )


def parse_xai_error(payload):
    error = payload.get('error') if isinstance(payload, dict) else None
    if isinstance(error, dict):
        error_type = (
            error.get('type')
            or error.get('code')
            or payload.get('code')
            or ''
        )
        message = error.get('message') or error.get('error') or ''
        return str(error_type), str(message)
    if isinstance(error, str):
        return str(payload.get('code') or payload.get('type') or ''), error
    return (
        str(payload.get('code') or payload.get('type') or ''),
        str(payload.get('message') or ''),
    )


def parse_xai_rate_limits(key, headers):
    for header, attribute in RATE_LIMIT_HEADERS.items():
        value = headers.get(header)
        if value is None:
            continue
        if attribute.endswith(('_limit', '_remaining')):
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
        setattr(key, attribute, value)

    retry_after = headers.get('retry-after')
    if retry_after is not None:
        key.retry_after = str(retry_after)


def is_invalid_xai_auth(status, error_type, message):
    lowered_type = error_type.lower()
    lowered_message = message.lower()
    if status == 401:
        return True
    if lowered_type in ('authentication_error', 'invalid_api_key', 'unauthorized'):
        return True
    return status == 400 and any(
        marker in lowered_message
        for marker in ('invalid api key', 'incorrect api key', 'invalid authorization')
    )


def classify_xai_access_error(status, error_type, message):
    lowered_type = error_type.lower()
    lowered_message = message.lower()
    if any(
        marker in lowered_message
        for marker in (
            'insufficient balance',
            'insufficient credit',
            'credit balance',
            'no credits',
            'spending limit',
        )
    ):
        return 'no_balance'
    if status == 402 or 'billing' in lowered_type:
        return 'billing_error'
    if any(
        marker in lowered_message
        for marker in ('team is blocked', 'api key is blocked', 'api key is disabled')
    ):
        return 'blocked'
    if status == 403:
        return 'permission_restricted'
    return ''


def model_error_can_fallback(message):
    lowered = message.lower()
    return any(
        marker in lowered
        for marker in ('model', 'not supported', 'not available')
    )


def order_xai_text_models(model_records):
    text_models = [item for item in model_records if is_xai_text_model(item)]

    def model_rank(item):
        model = item['id'].lower()
        if 'non-reasoning' in model:
            family_rank = 0
        elif model == 'grok-4.3':
            family_rank = 1
        elif model == 'grok-4.5':
            family_rank = 2
        elif 'reasoning' in model:
            family_rank = 3
        elif 'multi-agent' in model:
            family_rank = 4
        elif 'build' in model:
            family_rank = 5
        else:
            family_rank = 2
        price = int(item.get('prompt_text_token_price') or 0) + int(
            item.get('completion_text_token_price') or 0
        )
        return family_rank, price, model

    return [item['id'] for item in sorted(text_models, key=model_rank)]


def is_xai_text_model(model_record):
    model = model_record['id'].lower()
    if any(media in model for media in ('imagine', 'image', 'video', 'voice')):
        return False
    return (
        model_record.get('prompt_text_token_price') is not None
        or model_record.get('completion_text_token_price') is not None
    )


def xai_status_text(key):
    labels = {
        'usable': 'usable inference',
        'no_balance': 'no usable balance',
        'billing_error': 'billing blocked',
        'blocked': 'blocked or disabled',
        'permission_restricted': 'permission restricted',
        'rate_limited': 'rate limited',
        'no_text_models': 'no text models available',
        'no_usable_model': 'no usable text model found',
        'model_list_error': 'model listing failed',
        'api_error': 'temporary API error',
        'validation_error': 'validation unavailable',
        'probe_error': 'inference probe failed',
    }
    return labels.get(key.access_status, 'inference status unknown')


def format_xai_key(key):
    details = []
    if key.model:
        details.append(f'model - {key.model}')
    if key.models:
        details.append(
            f'available models - {len(key.text_models)} text / {len(key.models)} total'
        )
    details.append(xai_status_text(key))
    if key.full_endpoint_access and key.full_model_access:
        details.append('full ACL access')
    if key.usage_tokens:
        details.append(f'test tokens - {key.usage_tokens}')
    if key.test_cost_usd is not None:
        details.append(f'test cost - ${key.test_cost_usd:.8f}')
    if key.requests_limit:
        if key.requests_remaining is not None:
            details.append(
                f'request rate remaining - {key.requests_remaining}/{key.requests_limit}'
            )
        else:
            details.append(f'request rate limit - {key.requests_limit}')
    if key.tokens_limit:
        if key.tokens_remaining is not None:
            details.append(
                f'token rate remaining - {key.tokens_remaining}/{key.tokens_limit}'
            )
        else:
            details.append(f'token rate limit - {key.tokens_limit}')
    if key.retry_after:
        details.append(f'retry after - {key.retry_after}s')
    if key.has_usable_billing is None and key.error_type:
        details.append(f'error type - {key.error_type}')
    return f'{key.api_key} | ' + ' | '.join(details)


def pretty_print_xai_keys(keys):
    confirmed_keys = [key for key in keys if key.validation_status == 'valid']
    unavailable_keys = [key for key in keys if key.validation_status == 'unavailable']
    usable_keys = [
        key for key in confirmed_keys
        if key.has_usable_billing is True and not key.rate_limited
    ]
    no_balance_keys = [
        key for key in confirmed_keys
        if key.has_usable_billing is False
    ]
    blocked_keys = [key for key in confirmed_keys if key.blocked]
    rate_limited_keys = [key for key in confirmed_keys if key.rate_limited]
    permission_keys = [
        key for key in confirmed_keys
        if key.permission_restricted and not key.blocked and not key.rate_limited
    ]
    categorized_ids = {
        id(key)
        for group in (
            usable_keys,
            no_balance_keys,
            blocked_keys,
            rate_limited_keys,
            permission_keys,
        )
        for key in group
    }
    unknown_keys = [
        key for key in confirmed_keys
        if id(key) not in categorized_ids
    ]

    print('-' * 90)
    print(f'Validated {len(confirmed_keys)} live xAI keys:')
    print_xai_group('keys with usable inference', usable_keys)
    print_xai_group('keys without usable billing', no_balance_keys)
    print_xai_group('blocked or disabled keys', blocked_keys)
    print_xai_group('rate-limited keys', rate_limited_keys)
    print_xai_group('permission-restricted keys', permission_keys)
    print_xai_group('live keys with unknown inference status', unknown_keys)
    print_xai_group('keys that could not be verified', unavailable_keys)

    print(
        f'\n--- Total Live xAI Keys: {len(confirmed_keys)} '
        f'({len(usable_keys)} usable, {len(no_balance_keys)} without billing, '
        f'{len(blocked_keys)} blocked, {len(rate_limited_keys)} rate limited, '
        f'{len(permission_keys)} permission restricted, {len(unknown_keys)} unknown)'
        f'{f" | {len(unavailable_keys)} could not be verified" if unavailable_keys else ""} '
        f'---\n'
    )


def print_xai_group(label, keys):
    if not keys:
        return
    print(f'\n{len(keys)} {label}:')
    for key in keys:
        print(format_xai_key(key))
