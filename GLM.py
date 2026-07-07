import APIKey


GLM_API_BASE = 'https://open.bigmodel.cn/api/paas/v4'
GLM_BALANCE_ERROR_CODE = '1113'
GLM_BALANCE_CHECK_MODELS = (
    'glm-5-turbo',
    'glm-4.5',
    'glm-4.6',
    'glm-5',
)
GLM_AIR_CHECK_MODELS = (
    'glm-4.5-air',
)


async def response_json(response):
    try:
        return await response.json()
    except Exception:
        return {}


def choose_glm_model(models):
    model_ids = [model.get('id', '') for model in models]
    for model in GLM_BALANCE_CHECK_MODELS:
        if model in model_ids:
            return model
    return model_ids[0] if model_ids else GLM_BALANCE_CHECK_MODELS[0]


def choose_glm_air_model(models):
    model_ids = [model.get('id', '') for model in models]
    for model in GLM_AIR_CHECK_MODELS:
        if model in model_ids:
            return model
    return ''


def format_glm_error_message(error_code, message):
    if error_code == GLM_BALANCE_ERROR_CODE:
        return 'insufficient balance or no resource package'
    return str(message).encode('ascii', errors='backslashreplace').decode('ascii')


async def check_glm(key: APIKey, session):
    headers = {'Authorization': f'Bearer {key.api_key}', 'Content-Type': 'application/json'}
    async with session.get(f'{GLM_API_BASE}/models', headers=headers) as response:
        if response.status in (401, 403):
            return
        if response.status != 200:
            return
        data = await response_json(response)
        key.models = data.get('data', [])
        key.model = choose_glm_model(key.models)
        key.air_model = choose_glm_air_model(key.models)

    return await check_glm_balance(key, session, headers)


async def check_glm_balance(key: APIKey, session, headers):
    data = {
        'model': key.model,
        'messages': [{'role': 'user', 'content': 'ping'}],
        'max_tokens': 1,
    }
    async with session.post(f'{GLM_API_BASE}/chat/completions', headers=headers, json=data) as response:
        resp_json = await response_json(response)
        error = resp_json.get('error', {})
        key.error_code = str(error.get('code', ''))
        key.error_message = format_glm_error_message(key.error_code, error.get('message', ''))

        if response.status == 200:
            key.has_balance = True
            usage = resp_json.get('usage', {})
            key.usage_tokens = usage.get('total_tokens', 0)
            return True
        if key.error_code == GLM_BALANCE_ERROR_CODE:
            key.has_balance = False
            await check_glm_air_access(key, session, headers)
            return True
        if response.status == 429:
            key.rate_limited = True
            return True
        return response.status not in (401, 403)


async def check_glm_air_access(key: APIKey, session, headers):
    if not key.air_model:
        return
    data = {
        'model': key.air_model,
        'messages': [{'role': 'user', 'content': 'ping'}],
        'max_tokens': 1,
    }
    async with session.post(f'{GLM_API_BASE}/chat/completions', headers=headers, json=data) as response:
        resp_json = await response_json(response)
        if response.status == 200:
            key.has_air_access = True
            usage = resp_json.get('usage', {})
            key.air_usage_tokens = usage.get('total_tokens', 0)


def pretty_print_glm_keys(keys):
    print('-' * 90)
    keys_with_balance = [key for key in keys if key.has_balance and not key.rate_limited]
    keys_without_balance_with_air = [key for key in keys if not key.has_balance and key.has_air_access]
    keys_without_balance = [key for key in keys if not key.has_balance and not key.has_air_access]
    rate_limited_keys = [key for key in keys if key.rate_limited]

    print(f'Validated {len(keys)} GLM keys:')
    if keys_with_balance:
        print(f'\n{len(keys_with_balance)} keys with balance:')
        for key in keys_with_balance:
            usage = f' | test tokens - {key.usage_tokens}' if key.usage_tokens else ''
            print(f'{key.api_key} | model - {key.model}{usage}')
    if keys_without_balance_with_air:
        print(f'\n{len(keys_without_balance_with_air)} keys without balance but with air access:')
        for key in keys_without_balance_with_air:
            reason = f' | error {key.error_code} - {key.error_message}' if key.error_code else ''
            usage = f' | air test tokens - {key.air_usage_tokens}' if key.air_usage_tokens else ''
            print(f'{key.api_key} | paid model - {key.model} | air model - {key.air_model}{usage}{reason}')
    if keys_without_balance:
        print(f'\n{len(keys_without_balance)} keys without balance and without air access:')
        for key in keys_without_balance:
            reason = f' | error {key.error_code} - {key.error_message}' if key.error_code else ''
            print(f'{key.api_key} | model - {key.model}{reason}')
    if rate_limited_keys:
        print(f'\n{len(rate_limited_keys)} rate-limited keys:')
        for key in rate_limited_keys:
            reason = f' | error {key.error_code} - {key.error_message}' if key.error_code else ''
            print(f'{key.api_key} | model - {key.model}{reason}')

    print(f'\n--- Total Valid GLM Keys: {len(keys)} ({len(keys_with_balance)} with balance, {len(keys_without_balance_with_air)} without balance but with air access, {len(keys_without_balance)} without balance and without air access) ---\n')
