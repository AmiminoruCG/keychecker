import APIKey


MOONSHOT_API_BASE = 'https://api.moonshot.ai/v1'
MOONSHOT_KIMI_MODEL = 'kimi-k2.6'


async def response_json(response):
    try:
        return await response.json()
    except Exception:
        return {}


async def check_moonshot(key: APIKey, session):
    headers = {'Authorization': f'Bearer {key.api_key}', 'Content-Type': 'application/json'}
    if await check_moonshot_balance(key, session, headers) is None:
        return
    if await retrieve_moonshot_models(key, session, headers) is None:
        return
    if key.has_balance and key.has_kimi_k26:
        await test_moonshot_kimi(key, session, headers)
    return True


async def check_moonshot_balance(key: APIKey, session, headers):
    async with session.get(f'{MOONSHOT_API_BASE}/users/me/balance', headers=headers) as response:
        if response.status in (401, 403):
            return
        if response.status != 200:
            return
        data = await response_json(response)
        balance = data.get('data', {})
        key.available_balance = float(balance.get('available_balance', 0) or 0)
        key.cash_balance = float(balance.get('cash_balance', 0) or 0)
        key.voucher_balance = float(balance.get('voucher_balance', 0) or 0)
        key.has_balance = key.available_balance > 0
        return True


async def retrieve_moonshot_models(key: APIKey, session, headers):
    async with session.get(f'{MOONSHOT_API_BASE}/models', headers=headers) as response:
        if response.status in (401, 403):
            return
        if response.status != 200:
            return
        data = await response_json(response)
        key.models = [model.get('id', '') for model in data.get('data', [])]
        key.has_kimi_k26 = MOONSHOT_KIMI_MODEL in key.models
        return True


async def test_moonshot_kimi(key: APIKey, session, headers):
    data = {
        'model': MOONSHOT_KIMI_MODEL,
        'messages': [{'role': 'user', 'content': 'ping'}],
        'max_completion_tokens': 1,
    }
    async with session.post(f'{MOONSHOT_API_BASE}/chat/completions', headers=headers, json=data) as response:
        resp_json = await response_json(response)
        error = resp_json.get('error', {})
        key.error_message = error.get('message', '')
        key.error_type = error.get('type', '')

        if response.status == 200:
            key.kimi_k26_usable = True
            usage = resp_json.get('usage', {})
            key.usage_tokens = usage.get('total_tokens', 0)
        elif response.status == 429:
            key.rate_limited = True


def moonshot_balance_details(key):
    return f'balance - ${key.available_balance:.6f} (cash ${key.cash_balance:.6f}, voucher ${key.voucher_balance:.6f})'


def pretty_print_moonshot_keys(keys):
    print('-' * 90)
    keys_with_balance = [key for key in keys if key.has_balance]
    keys_without_balance = [key for key in keys if not key.has_balance]

    print(f'Validated {len(keys)} Moonshot keys:')
    if keys_with_balance:
        print(f'\n{len(keys_with_balance)} keys with balance:')
        for key in keys_with_balance:
            kimi = ' | has kimi-k2.6' if key.has_kimi_k26 else ''
            usable = ' | kimi-k2.6 usable' if key.kimi_k26_usable else ''
            usage = f' | test tokens - {key.usage_tokens}' if key.usage_tokens else ''
            print(f'{key.api_key} | {moonshot_balance_details(key)}{kimi}{usable}{usage}')
    if keys_without_balance:
        print(f'\n{len(keys_without_balance)} keys without balance:')
        for key in keys_without_balance:
            kimi = ' | has kimi-k2.6' if key.has_kimi_k26 else ''
            print(f'{key.api_key} | {moonshot_balance_details(key)}{kimi}')

    print(f'\n--- Total Valid Moonshot Keys: {len(keys)} ({len(keys_with_balance)} with balance, {len(keys_without_balance)} without balance) ---\n')
