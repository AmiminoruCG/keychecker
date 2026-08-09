import unittest

from Anthropic import check_anthropic, order_anthropic_models
from APIKey import APIKey, Provider


class FakeResponse:
    def __init__(self, status, payload, headers=None):
        self.status = status
        self.payload = payload
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def json(self, **kwargs):
        return self.payload


class FakeSession:
    def __init__(self, get_responses, post_responses=None):
        self.get_responses = list(get_responses)
        self.post_responses = list(post_responses or [])

    def get(self, *args, **kwargs):
        return self.get_responses.pop(0)

    def post(self, *args, **kwargs):
        return self.post_responses.pop(0)


def models_response(*models):
    return FakeResponse(200, {'data': [{'id': model} for model in models]})


class AnthropicCheckerTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_key_with_usable_balance(self):
        session = FakeSession(
            [models_response('claude-opus-5', 'claude-haiku-4-5-20251001')],
            [FakeResponse(
                200,
                {'usage': {'input_tokens': 8, 'output_tokens': 1}},
                {
                    'anthropic-ratelimit-requests-limit': '1000',
                    'anthropic-ratelimit-requests-remaining': '999',
                    'anthropic-ratelimit-input-tokens-limit': '100000',
                    'anthropic-ratelimit-input-tokens-remaining': '99000',
                },
            )],
        )
        key = APIKey(Provider.ANTHROPIC, 'test-key')

        result = await check_anthropic(key, session)

        self.assertTrue(result)
        self.assertEqual(key.validation_status, 'valid')
        self.assertIs(key.has_quota, True)
        self.assertEqual(key.quota_status, 'available')
        self.assertEqual(key.model, 'claude-haiku-4-5-20251001')
        self.assertEqual(key.usage_tokens, 9)
        self.assertEqual(key.requests_remaining, 999)

    async def test_live_key_without_balance(self):
        session = FakeSession(
            [models_response('claude-haiku-4-5-20251001')],
            [FakeResponse(400, {
                'type': 'error',
                'error': {
                    'type': 'invalid_request_error',
                    'message': (
                        'Your credit balance is too low to access the Anthropic API. '
                        'Please purchase credits.'
                    ),
                },
            })],
        )
        key = APIKey(Provider.ANTHROPIC, 'test-key')

        result = await check_anthropic(key, session)

        self.assertTrue(result)
        self.assertEqual(key.validation_status, 'valid')
        self.assertIs(key.has_quota, False)
        self.assertEqual(key.quota_status, 'no_balance')

    async def test_invalid_key(self):
        session = FakeSession([FakeResponse(401, {
            'type': 'error',
            'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'},
        })])
        key = APIKey(Provider.ANTHROPIC, 'test-key')

        result = await check_anthropic(key, session)

        self.assertIsNone(result)
        self.assertEqual(key.validation_status, 'invalid')

    async def test_rate_limited_key_is_live_but_quota_is_unknown(self):
        session = FakeSession(
            [models_response('claude-haiku-4-5-20251001')],
            [FakeResponse(
                429,
                {
                    'type': 'error',
                    'error': {'type': 'rate_limit_error', 'message': 'rate limited'},
                },
                {'retry-after': '12'},
            )],
        )
        key = APIKey(Provider.ANTHROPIC, 'test-key')

        result = await check_anthropic(key, session)

        self.assertTrue(result)
        self.assertTrue(key.rate_limited)
        self.assertIsNone(key.has_quota)
        self.assertEqual(key.retry_after, '12')

    async def test_generic_bad_request_does_not_claim_quota(self):
        session = FakeSession(
            [models_response('claude-haiku-4-5-20251001')],
            [FakeResponse(400, {
                'type': 'error',
                'error': {'type': 'invalid_request_error', 'message': 'unexpected field'},
            })],
        )
        key = APIKey(Provider.ANTHROPIC, 'test-key')

        result = await check_anthropic(key, session)

        self.assertTrue(result)
        self.assertIsNone(key.has_quota)
        self.assertEqual(key.quota_status, 'probe_error')

    async def test_retired_model_falls_back_to_another_listed_model(self):
        session = FakeSession(
            [models_response('claude-haiku-old', 'claude-sonnet-5')],
            [
                FakeResponse(404, {
                    'type': 'error',
                    'error': {'type': 'not_found_error', 'message': 'model not found'},
                }),
                FakeResponse(200, {'usage': {'input_tokens': 6, 'output_tokens': 1}}),
            ],
        )
        key = APIKey(Provider.ANTHROPIC, 'test-key')

        result = await check_anthropic(key, session)

        self.assertTrue(result)
        self.assertIs(key.has_quota, True)
        self.assertEqual(key.model, 'claude-sonnet-5')

    async def test_api_outage_is_not_reported_as_invalid(self):
        session = FakeSession([FakeResponse(529, {
            'type': 'error',
            'error': {'type': 'overloaded_error', 'message': 'overloaded'},
        })])
        key = APIKey(Provider.ANTHROPIC, 'test-key')

        result = await check_anthropic(key, session)

        self.assertFalse(result)
        self.assertEqual(key.validation_status, 'unavailable')

    def test_model_order_prefers_haiku(self):
        models = ['claude-opus-5', 'claude-sonnet-5', 'claude-haiku-4-5']
        self.assertEqual(
            order_anthropic_models(models),
            ['claude-haiku-4-5', 'claude-sonnet-5', 'claude-opus-5'],
        )


if __name__ == '__main__':
    unittest.main()
