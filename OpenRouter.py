import asyncio
import math

import aiohttp

from APIKey import APIKey

OPENROUTER_API_BASE = 'https://openrouter.ai/api/v1'
OPENROUTER_TIMEOUT = aiohttp.ClientTimeout(total=20, connect=10, sock_read=10)


async def response_json(response):
    try:
        payload = await response.json()
    except (ValueError, aiohttp.ContentTypeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def parse_openrouter_number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


async def check_openrouter(key: APIKey, session):
    # A repeated check must not retain a previous balance or authentication result.
    key.__dict__.update(APIKey(key.provider, key.api_key).__dict__)
    headers = {'Authorization': f'Bearer {key.api_key}'}
    try:
        async with session.get(
            f'{OPENROUTER_API_BASE}/key', headers=headers,
            timeout=OPENROUTER_TIMEOUT, allow_redirects=False,
        ) as response:
            if response.status in (401, 403):
                key.validation_status = 'invalid'
                key.validation_message = f'authentication rejected (HTTP {response.status})'
                return None
            if response.status != 200:
                key.validation_message = f'key API returned HTTP {response.status}; retry later'
                return True
            resp_json = await response_json(response)
            data = resp_json.get('data')
            if not isinstance(data, dict) or not any(
                field in data for field in ('label', 'usage', 'limit', 'is_free_tier')
            ):
                key.validation_message = 'invalid response from key API; retry later'
                return True
            parse_openrouter_key_info(key, data)
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
        key.validation_message = 'key API network error or timeout; retry later'
        return True

    key.validation_status = 'valid'
    key.validation_message = 'authenticated; model requests not tested'
    await get_openrouter_credits(key, session, headers)
    return True


def parse_openrouter_key_info(key: APIKey, data):
    for field in ('usage', 'usage_daily', 'usage_weekly', 'usage_monthly'):
        setattr(key, field, parse_openrouter_number(data.get(field)))

    key.credit_limit = parse_openrouter_number(data.get('limit'))
    key.key_limit_known = 'limit' in data and (
        data['limit'] is None or key.credit_limit is not None
    )
    key.limit_remaining = parse_openrouter_number(data.get('limit_remaining'))
    key.limit_reset = data.get('limit_reset')
    key.limit_reset_known = 'limit_reset' in data
    free_tier = data.get('is_free_tier')
    key.bought_credits = not free_tier if isinstance(free_tier, bool) else None
    key.include_byok_in_limit = data.get('include_byok_in_limit')
    key.is_management_key = data.get('is_management_key', data.get('is_provisioning_key'))
    has_cap_or_unknown = not key.key_limit_known or key.credit_limit is not None
    key.limit_reached = has_cap_or_unknown and (
        (key.credit_limit is not None and key.credit_limit <= 0)
        or (key.limit_remaining is not None and key.limit_remaining <= 0)
    )
    # The API's rate_limit object is deprecated and does not report current RPM.
    key.rpm = 0


async def get_openrouter_credits(key: APIKey, session, headers):
    key.account_balance = key.balance = None
    key.total_credits = key.total_usage = None
    key.credits_api_available = False
    key.credits_api_status = 'not_checked'
    key.credits_api_http_status = None
    try:
        async with session.get(
            f'{OPENROUTER_API_BASE}/credits', headers=headers,
            timeout=OPENROUTER_TIMEOUT, allow_redirects=False,
        ) as response:
            key.credits_api_http_status = response.status
            if response.status != 200:
                key.credits_api_status = (
                    'permission_denied' if response.status in (401, 403)
                    else 'rate_limited' if response.status == 429
                    else 'http_error'
                )
                return
            resp_json = await response_json(response)
            data = resp_json.get('data')
            if not isinstance(data, dict):
                key.credits_api_status = 'invalid_response'
                return
            total_credits = parse_openrouter_number(data.get('total_credits'))
            total_usage = parse_openrouter_number(data.get('total_usage'))
            if total_credits is None or total_usage is None:
                key.credits_api_status = 'invalid_response'
                return
            balance = total_credits - total_usage
            if not math.isfinite(balance):
                key.credits_api_status = 'invalid_response'
                return
            key.total_credits = total_credits
            key.total_usage = total_usage
            key.account_balance = key.balance = balance
            key.credits_api_available = True
            key.credits_api_status = 'ok'
    except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
        # Account metadata is optional; losing it must not discard a valid key.
        key.credits_api_status = 'network_error'


def get_openrouter_spending_budget(key):
    """Upper bound from the two reported caps, not proof of model access."""
    if key.validation_status != 'valid':
        return None
    if key.limit_reached or (key.account_balance is not None and key.account_balance <= 0):
        return 0.0
    if key.account_balance is None or not key.key_limit_known:
        return None
    if key.credit_limit is None:
        return key.account_balance
    if key.limit_remaining is None:
        return None
    return max(0.0, min(key.account_balance, key.limit_remaining, key.credit_limit))


def format_openrouter_money(value):
    if value is None:
        return 'unknown'
    if 0 < abs(value) < 0.0001:
        return '<$0.0001' if value > 0 else '-<$0.0001'
    return f'${value:,.4f}'


def openrouter_status_text(key):
    if key.validation_status == 'valid':
        return 'authenticated (model requests not tested)'
    return f'{key.validation_status}: {key.validation_message}'


def openrouter_balance_unavailable_reason(key):
    status = key.credits_api_status
    if status == 'permission_denied':
        if key.credits_api_http_status == 403:
            return 'not exposed to this key; /credits requires a management key (HTTP 403)'
        return f'credits API access denied (HTTP {key.credits_api_http_status})'
    if status == 'rate_limited':
        return 'credits API rate limited (HTTP 429); retry later'
    if status == 'http_error':
        return f'credits API returned HTTP {key.credits_api_http_status}; retry later'
    if status == 'network_error':
        return 'credits API network error or timeout; retry later'
    if status == 'invalid_response':
        return 'credits API returned missing or invalid amounts'
    return 'account credits not checked'


def format_openrouter_key(key):
    lines = [key.api_key, f'  Status: {openrouter_status_text(key)}']
    if key.validation_status != 'valid':
        lines.append('  Account balance: unknown (authentication not confirmed)')
        return '\n'.join(lines)

    budget = get_openrouter_spending_budget(key)
    if budget is None:
        reason = 'account balance unavailable' if key.account_balance is None else 'key limit data incomplete'
        budget_note = f' ({reason})'
    elif key.limit_reached:
        budget_note = ' (key spending limit exhausted)'
    elif key.account_balance is not None and key.account_balance <= 0:
        budget_note = ' (no account credits available)'
    else:
        budget_note = ' (upper bound from these two limits)'
    lines.append(f'  Budget from balance/key limit: {format_openrouter_money(budget)}{budget_note}')

    if key.account_balance is None:
        lines.append(f'  Account balance: unknown ({openrouter_balance_unavailable_reason(key)})')
    else:
        lines.append(f'  Account balance: {format_openrouter_money(key.account_balance)} (shared across the account)')

    if not key.key_limit_known:
        lines.append('  Key spending limit: unknown (not reported by API)')
    elif key.credit_limit is None:
        lines.append('  Key spending limit: none (no per-key cap; account balance still applies)')
    else:
        lines.append(f'  Key spending limit: {format_openrouter_money(key.credit_limit)}')
    if not key.key_limit_known or key.credit_limit is not None:
        lines.append(f'  Remaining key limit: {format_openrouter_money(key.limit_remaining)} (spending allowance, not account balance)')
        reset_names = {'daily': 'daily (UTC)', 'weekly': 'weekly (UTC)', 'monthly': 'monthly (UTC)'}
        if not key.limit_reset_known:
            reset = 'unknown'
        elif key.limit_reset is None:
            reset = 'never (lifetime limit)'
        else:
            reset = reset_names.get(str(key.limit_reset), str(key.limit_reset))
        lines.append(f'  Key limit resets: {reset}')
        if key.include_byok_in_limit is True:
            lines.append('  Key limit also counts usage with your own provider keys (BYOK).')

    lines.append(f'  Spent with this key: {format_openrouter_money(key.usage)} total')
    lines.append(
        f'    Current UTC day: {format_openrouter_money(key.usage_daily)}'
        f' | week: {format_openrouter_money(key.usage_weekly)}'
        f' | month: {format_openrouter_money(key.usage_monthly)}'
    )
    if key.credits_api_available:
        lines.append(
            f'  Account totals: {format_openrouter_money(key.total_credits)} credits purchased'
            f' - {format_openrouter_money(key.total_usage)} spent = {format_openrouter_money(key.account_balance)} balance'
        )
    if key.is_management_key is True:
        lines.append('  Key type: management (for account administration, not model requests)')
    return '\n'.join(lines)


def pretty_print_openrouter_keys(keys):
    print('-' * 90)
    valid = [key for key in keys if key.validation_status == 'valid']
    inconclusive = [key for key in keys if key.validation_status == 'unknown']
    print(f'OpenRouter: {len(valid)} authenticated, {len(inconclusive)} inconclusive.')
    print('All amounts are USD. Budget considers account balance and the key spending limit only.')
    print('Model access and free-model request limits are not tested.')
    for key in sorted(keys, key=lambda item: item.api_key):
        print('\n' + format_openrouter_key(key))
    budgets = [get_openrouter_spending_budget(key) for key in valid]
    positive = sum(value is not None and value > 0 for value in budgets)
    zero = sum(value == 0 for value in budgets)
    unknown = sum(value is None for value in budgets)
    print(
        f'\n--- OpenRouter: {positive} with positive budget, {zero} with zero budget, '
        f'{unknown} with unknown budget; {len(inconclusive)} inconclusive checks ---\n'
    )
