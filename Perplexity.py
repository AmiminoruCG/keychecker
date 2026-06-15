import APIKey


async def check_perplexity(key: APIKey, session):
    headers = {'Authorization': f'Bearer {key.api_key}', 'Content-Type': 'application/json'}
    data = {
        "model": "perplexity/sonar",
        "input": "ping",
        "max_output_tokens": 1,
    }
    async with session.post('https://api.perplexity.ai/v1/agent', headers=headers, json=data) as response:
        if response.status in (401, 403):
            return
        if response.status == 429:
            key.rate_limited = True
            return True
        if response.status == 402:
            key.has_quota = False
            return True
        if response.status != 200:
            return

        data = await response.json()
        key.model = data.get('model', 'sonar')
        usage = data.get('usage', {})
        key.usage_tokens = usage.get('total_tokens', 0)
        cost = usage.get('cost', {})
        key.cost = cost.get('total_cost', 0)
        return True


def pretty_print_perplexity_keys(keys):
    print('-' * 90)
    keys_with_quota = [key for key in keys if key.has_quota and not key.rate_limited]
    keys_without_quota = [key for key in keys if not key.has_quota]
    rate_limited_keys = [key for key in keys if key.rate_limited]

    print(f'Validated {len(keys)} Perplexity keys:')
    if keys_with_quota:
        print(f'\n{len(keys_with_quota)} keys with quota:')
        for key in keys_with_quota:
            cost = f' | test cost - ${format(key.cost, ".6f")}' if key.cost else ''
            usage = f' | test tokens - {key.usage_tokens}' if key.usage_tokens else ''
            print(f'{key.api_key} | model - {key.model}{usage}{cost}')
    if keys_without_quota:
        print(f'\n{len(keys_without_quota)} keys without quota:')
        for key in keys_without_quota:
            print(f'{key.api_key}')
    if rate_limited_keys:
        print(f'\n{len(rate_limited_keys)} rate-limited keys:')
        for key in rate_limited_keys:
            print(f'{key.api_key}')

    print(f'\n--- Total Valid Perplexity Keys: {len(keys)} ({len(keys_with_quota)} with quota) ---\n')
