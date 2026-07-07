import APIKey

OPENROUTER_API_BASE = 'https://openrouter.ai/api/v1'


async def response_json(response):
    try:
        return await response.json()
    except Exception:
        return {}


async def check_openrouter(key: APIKey, session):
    headers = {'Authorization': f'Bearer {key.api_key}'}
    async with session.get(f'{OPENROUTER_API_BASE}/key', headers=headers) as response:
        if response.status != 200:
            return
        resp_json = await response_json(response)
        data = resp_json.get('data')
        if data is None:
            return

        parse_openrouter_key_info(key, data)
        await get_openrouter_credits(key, session, headers)

        return True


def parse_openrouter_key_info(key: APIKey, data):
    key.usage = float(data.get('usage') or 0)
    key.usage_daily = float(data.get('usage_daily') or 0)
    key.usage_weekly = float(data.get('usage_weekly') or 0)
    key.usage_monthly = float(data.get('usage_monthly') or 0)
    key.credit_limit = data.get('limit')
    key.limit_remaining = data.get('limit_remaining')
    key.limit_reset = data.get('limit_reset')
    key.bought_credits = not data.get('is_free_tier', True)
    key.limit_reached = key.limit_remaining is not None and key.limit_remaining <= 0

    rate_limit = data.get('rate_limit') or {}
    if rate_limit.get('requests') and rate_limit.get('interval'):
        key.rpm = parse_openrouter_rpm(rate_limit)


def parse_openrouter_rpm(rate_limit):
    interval = str(rate_limit.get('interval', ''))
    try:
        requests = int(rate_limit.get('requests', 0))
    except (TypeError, ValueError):
        return 0
    if requests <= 0:
        return 0
    try:
        if interval.endswith('ms'):
            seconds = int(interval[:-2]) / 1000
        elif interval.endswith('s'):
            seconds = int(interval[:-1])
        elif interval.endswith('m'):
            seconds = int(interval[:-1]) * 60
        elif interval.endswith('h'):
            seconds = int(interval[:-1]) * 3600
        else:
            seconds = 60
    except (TypeError, ValueError):
        return 0
    return int(requests / seconds * 60) if seconds > 0 else 0


async def get_openrouter_credits(key: APIKey, session, headers):
    async with session.get(f'{OPENROUTER_API_BASE}/credits', headers=headers) as response:
        if response.status != 200:
            return
        resp_json = await response_json(response)
        data = resp_json.get('data') or {}
        key.total_credits = float(data.get('total_credits') or 0)
        key.total_usage = float(data.get('total_usage') or 0)
        key.account_balance = key.total_credits - key.total_usage
        key.balance = key.account_balance
        key.credits_api_available = True


def format_openrouter_money(value):
    return 'unlimited' if value is None else f'${format(float(value), ".4f")}'


def pretty_print_openrouter_keys(keys):
    print('-' * 90)
    keys_with_balance = [key for key in keys if key.account_balance is not None and key.account_balance > 0]
    keys_without_balance = [key for key in keys if key.account_balance is not None and key.account_balance <= 0]
    keys_unknown_balance = [key for key in keys if key.account_balance is None]

    print(f'Validated {len(keys)} OpenRouter keys:')
    if keys_with_balance:
        print(f'\n{len(keys_with_balance)} keys with account balance:')
        for key in keys_with_balance:
            print(format_openrouter_key(key))

    if keys_without_balance:
        print(f'\n{len(keys_without_balance)} keys without account balance:')
        for key in keys_without_balance:
            print(format_openrouter_key(key))

    if keys_unknown_balance:
        print(f'\n{len(keys_unknown_balance)} keys with unknown account balance:')
        for key in keys_unknown_balance:
            print(format_openrouter_key(key) + ' | credits API unavailable')

    print(f'\n--- Total Valid OpenRouter Keys: {len(keys)} ({len(keys_with_balance)} with account balance, {len(keys_without_balance)} without account balance, {len(keys_unknown_balance)} unknown balance) ---\n')


def format_openrouter_key(key):
    details = [key.api_key]
    if key.account_balance is not None:
        details.append(f'account balance - {format_openrouter_money(key.account_balance)}')
        details.append(f'total credits - {format_openrouter_money(key.total_credits)}')
        details.append(f'total usage - {format_openrouter_money(key.total_usage)}')
    details.append(f'key usage - {format_openrouter_money(key.usage)}')
    details.append(f'daily - {format_openrouter_money(key.usage_daily)}')
    details.append(f'weekly - {format_openrouter_money(key.usage_weekly)}')
    details.append(f'monthly - {format_openrouter_money(key.usage_monthly)}')
    if key.credit_limit is not None:
        details.append(f'key limit - {format_openrouter_money(key.credit_limit)}')
        details.append(f'key remaining - {format_openrouter_money(key.limit_remaining)}')
        if key.limit_reset:
            details.append(f'limit reset - {key.limit_reset}')
    else:
        details.append('key limit - unlimited')
    if key.rpm > 0:
        details.append(f'RPM - {key.rpm}')
    if key.limit_reached:
        details.append('LIMIT REACHED')
    if key.bought_credits:
        details.append('purchased credits')
    return ' | '.join(details)
