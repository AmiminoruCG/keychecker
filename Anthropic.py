import APIKey


ANTHROPIC_API_BASE = 'https://api.anthropic.com/v1'
ANTHROPIC_VERSION = '2023-06-01'
MODEL_FAMILY_PREFERENCE = ('haiku', 'fable', 'sonnet', 'opus')

RATE_LIMIT_HEADERS = {
    'anthropic-ratelimit-requests-limit': 'requests_limit',
    'anthropic-ratelimit-requests-remaining': 'requests_remaining',
    'anthropic-ratelimit-requests-reset': 'requests_reset',
    'anthropic-ratelimit-tokens-limit': 'tokens_limit',
    'anthropic-ratelimit-tokens-remaining': 'tokens_remaining',
    'anthropic-ratelimit-tokens-reset': 'tokens_reset',
    'anthropic-ratelimit-input-tokens-limit': 'input_tokens_limit',
    'anthropic-ratelimit-input-tokens-remaining': 'input_tokens_remaining',
    'anthropic-ratelimit-input-tokens-reset': 'input_tokens_reset',
    'anthropic-ratelimit-output-tokens-limit': 'output_tokens_limit',
    'anthropic-ratelimit-output-tokens-remaining': 'output_tokens_remaining',
    'anthropic-ratelimit-output-tokens-reset': 'output_tokens_reset',
}


async def check_anthropic(key: APIKey, session):
    headers = {
        'anthropic-version': ANTHROPIC_VERSION,
        'x-api-key': key.api_key,
    }

    async with session.get(
        f'{ANTHROPIC_API_BASE}/models',
        headers=headers,
        params={'limit': 100},
    ) as response:
        payload = await response_json(response)
        capture_anthropic_response(key, response, payload)

        if response.status == 401:
            key.validation_status = 'invalid'
            return None
        if response.status == 402:
            mark_anthropic_no_quota(key, 'billing_error')
            key.validation_status = 'valid'
            return True
        if response.status == 403:
            key.validation_status = 'valid'
            key.permission_restricted = True
            key.quota_status = 'permission_restricted'
            return True
        if response.status == 429:
            key.validation_status = 'valid'
            key.rate_limited = True
            key.quota_status = 'rate_limited'
            return True
        if response.status >= 500:
            key.validation_status = 'unavailable'
            key.quota_status = 'api_error'
            return False
        if response.status != 200:
            key.validation_status = 'unavailable'
            key.quota_status = 'validation_error'
            return False

        key.validation_status = 'valid'
        key.models = [
            item.get('id')
            for item in payload.get('data', [])
            if isinstance(item, dict) and item.get('id')
        ]

    if not key.models:
        key.quota_status = 'no_models'
        return True

    candidates = order_anthropic_models(key.models)[:3]
    for index, model in enumerate(candidates):
        key.model = model
        data = {
            'model': model,
            'messages': [{'role': 'user', 'content': 'Reply with OK.'}],
            'max_tokens': 1,
        }
        message_headers = {**headers, 'content-type': 'application/json'}

        async with session.post(
            f'{ANTHROPIC_API_BASE}/messages',
            headers=message_headers,
            json=data,
        ) as response:
            payload = await response_json(response)
            capture_anthropic_response(key, response, payload)

            if response.status == 200:
                usage = payload.get('usage', {})
                key.input_tokens_used = int(usage.get('input_tokens') or 0)
                key.output_tokens_used = int(usage.get('output_tokens') or 0)
                key.usage_tokens = sum(
                    int(usage.get(field) or 0)
                    for field in (
                        'input_tokens',
                        'output_tokens',
                        'cache_creation_input_tokens',
                        'cache_read_input_tokens',
                    )
                )
                key.has_quota = True
                key.quota_status = 'available'
                key.error_type = ''
                key.error_message = ''
                return True

            if response.status == 401:
                key.validation_status = 'invalid'
                return None

            quota_status = classify_anthropic_quota_error(
                response.status,
                key.error_type,
                key.error_message,
            )
            if quota_status:
                mark_anthropic_no_quota(key, quota_status)
                return True

            if response.status == 429:
                key.rate_limited = True
                key.quota_status = 'rate_limited'
                return True

            if response.status in (403, 404) and index + 1 < len(candidates):
                continue

            if response.status == 403:
                key.permission_restricted = True
                key.quota_status = 'permission_restricted'
            elif response.status == 404:
                key.quota_status = 'no_usable_model'
            elif response.status >= 500:
                key.quota_status = 'api_error'
            else:
                key.quota_status = 'probe_error'
            return True

    key.quota_status = 'no_usable_model'
    return True


async def response_json(response):
    try:
        payload = await response.json(content_type=None)
        return payload if isinstance(payload, dict) else {}
    except (TypeError, ValueError):
        return {}


def capture_anthropic_response(key, response, payload):
    parse_anthropic_rate_limits(key, response.headers)
    error = payload.get('error', {}) if isinstance(payload, dict) else {}
    key.error_type = str(error.get('type') or '')
    key.error_message = str(error.get('message') or '')
    key.request_id = str(
        response.headers.get('request-id')
        or payload.get('request_id')
        or ''
    )


def parse_anthropic_rate_limits(key, headers):
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
    key.rpm = key.requests_limit
    if key.tokens_remaining is not None:
        key.remaining_tokens = key.tokens_remaining


def order_anthropic_models(models):
    def model_rank(item):
        index, model = item
        lowered = model.lower()
        for rank, family in enumerate(MODEL_FAMILY_PREFERENCE):
            if family in lowered:
                return rank, index
        return len(MODEL_FAMILY_PREFERENCE), index

    return [model for _, model in sorted(enumerate(models), key=model_rank)]


def classify_anthropic_quota_error(status, error_type, message):
    lowered = message.lower()
    if 'organization has been disabled' in lowered:
        return 'organization_disabled'
    if (
        'credit balance is too low' in lowered
        or 'purchase credits' in lowered
        or 'insufficient credit' in lowered
    ):
        return 'no_balance'
    if (
        'specified api usage limits' in lowered
        or 'spend limit' in lowered
        or 'monthly limit' in lowered
    ):
        return 'spend_limit_reached'
    if status == 402 or error_type == 'billing_error':
        return 'billing_error'
    return ''


def mark_anthropic_no_quota(key, quota_status):
    key.has_quota = False
    key.quota_status = quota_status


def anthropic_status_text(key):
    labels = {
        'available': 'usable balance',
        'no_balance': 'no balance',
        'billing_error': 'billing blocked',
        'spend_limit_reached': 'spend limit reached',
        'organization_disabled': 'organization disabled',
        'rate_limited': 'rate limited',
        'permission_restricted': 'permission restricted',
        'no_models': 'no models available',
        'no_usable_model': 'no usable model found',
        'api_error': 'temporary API error',
        'validation_error': 'validation unavailable',
        'probe_error': 'balance probe failed',
    }
    return labels.get(key.quota_status, 'balance unknown')


def format_anthropic_key(key):
    details = []
    if key.model:
        details.append(f'model - {key.model}')
    if key.models:
        details.append(f'available models - {len(key.models)}')
    details.append(anthropic_status_text(key))
    if key.usage_tokens:
        details.append(f'test tokens - {key.usage_tokens}')
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
                f'combined token rate remaining - '
                f'{key.tokens_remaining}/{key.tokens_limit}'
            )
        else:
            details.append(f'combined token rate limit - {key.tokens_limit}')
    if key.input_tokens_limit:
        if key.input_tokens_remaining is not None:
            details.append(
                f'input token rate remaining - '
                f'{key.input_tokens_remaining}/{key.input_tokens_limit}'
            )
        else:
            details.append(f'input token rate limit - {key.input_tokens_limit}')
    if key.output_tokens_limit:
        if key.output_tokens_remaining is not None:
            details.append(
                f'output token rate remaining - '
                f'{key.output_tokens_remaining}/{key.output_tokens_limit}'
            )
        else:
            details.append(f'output token rate limit - {key.output_tokens_limit}')
    if key.retry_after:
        details.append(f'retry after - {key.retry_after}s')
    if key.has_quota is None and key.error_type:
        details.append(f'error type - {key.error_type}')
    return f'{key.api_key} | ' + ' | '.join(details)


def pretty_print_anthropic_keys(keys):
    confirmed_keys = [key for key in keys if key.validation_status == 'valid']
    unavailable_keys = [key for key in keys if key.validation_status == 'unavailable']
    keys_with_quota = [
        key for key in confirmed_keys
        if key.has_quota is True and not key.rate_limited
    ]
    keys_without_quota = [key for key in confirmed_keys if key.has_quota is False]
    rate_limited_keys = [key for key in confirmed_keys if key.rate_limited]
    unknown_quota_keys = [
        key for key in confirmed_keys
        if key.has_quota is None and not key.rate_limited
    ]

    print('-' * 90)
    print(f'Validated {len(confirmed_keys)} live Anthropic keys:')

    print_anthropic_group('keys with usable balance', keys_with_quota)
    print_anthropic_group('keys without usable balance', keys_without_quota)
    print_anthropic_group('rate-limited keys', rate_limited_keys)
    print_anthropic_group('live keys with unknown balance', unknown_quota_keys)
    print_anthropic_group('keys that could not be verified', unavailable_keys)

    print(
        f'\n--- Total Live Anthropic Keys: {len(confirmed_keys)} '
        f'({len(keys_with_quota)} with usable balance, '
        f'{len(keys_without_quota)} without usable balance, '
        f'{len(rate_limited_keys)} rate limited, '
        f'{len(unknown_quota_keys)} unknown)'
        f'{f" | {len(unavailable_keys)} could not be verified" if unavailable_keys else ""} '
        f'---\n'
    )


def print_anthropic_group(label, keys):
    if not keys:
        return
    print(f'\n{len(keys)} {label}:')
    for key in keys:
        print(format_anthropic_key(key))
