import unittest

from APIKey import APIKey, Provider
from XAI import check_xai, order_xai_text_models


class FakeResponse:
    def __init__(self, status, payload=None, headers=None):
        self.status = status
        self.payload = payload or {}
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


def key_metadata(**overrides):
    payload = {
        'team_blocked': False,
        'api_key_blocked': False,
        'api_key_disabled': False,
        'acls': ['api-key:endpoint:*', 'api-key:model:*'],
    }
    payload.update(overrides)
    return FakeResponse(200, payload)


def models_response(*models):
    return FakeResponse(200, {'data': list(models)})


def text_model(model_id, prompt_price=12500, completion_price=25000):
    return {
        'id': model_id,
        'prompt_text_token_price': prompt_price,
        'completion_text_token_price': completion_price,
    }


class XaiCheckerTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_key_with_usable_inference(self):
        session = FakeSession(
            [
                key_metadata(),
                models_response(
                    text_model('grok-4.5', 20000, 60000),
                    text_model('grok-4.20-non-reasoning'),
                    {'id': 'grok-imagine-image'},
                ),
            ],
            [FakeResponse(
                200,
                {
                    'service_tier': 'default',
                    'usage': {
                        'input_tokens': 20,
                        'output_tokens': 1,
                        'total_tokens': 21,
                        'cost_in_usd_ticks': 12345,
                    },
                },
                {
                    'x-ratelimit-limit-requests': '8300',
                    'x-ratelimit-remaining-requests': '8299',
                    'x-ratelimit-limit-tokens': '53000000',
                    'x-ratelimit-remaining-tokens': '52999979',
                },
            )],
        )
        key = APIKey(Provider.XAI, 'test-key')

        result = await check_xai(key, session)

        self.assertTrue(result)
        self.assertEqual(key.validation_status, 'valid')
        self.assertIs(key.has_usable_billing, True)
        self.assertEqual(key.model, 'grok-4.20-non-reasoning')
        self.assertEqual(key.usage_tokens, 21)
        self.assertEqual(key.cost_usd_ticks, 12345)
        self.assertEqual(key.requests_remaining, 8299)

    async def test_live_key_without_balance(self):
        session = FakeSession(
            [key_metadata(), models_response(text_model('grok-4.3'))],
            [FakeResponse(403, {
                'code': 'insufficient_credit',
                'error': 'Insufficient credits. Please add funds.',
            })],
        )
        key = APIKey(Provider.XAI, 'test-key')

        result = await check_xai(key, session)

        self.assertTrue(result)
        self.assertIs(key.has_usable_billing, False)
        self.assertEqual(key.access_status, 'no_balance')

    async def test_invalid_key(self):
        session = FakeSession([FakeResponse(401, {
            'code': 'unauthorized',
            'error': 'Invalid API key',
        })])
        key = APIKey(Provider.XAI, 'test-key')

        result = await check_xai(key, session)

        self.assertIsNone(result)
        self.assertEqual(key.validation_status, 'invalid')

    async def test_blocked_key_is_reported_separately(self):
        session = FakeSession([key_metadata(team_blocked=True)])
        key = APIKey(Provider.XAI, 'test-key')

        result = await check_xai(key, session)

        self.assertTrue(result)
        self.assertTrue(key.blocked)
        self.assertEqual(key.access_status, 'blocked')
        self.assertIsNone(key.has_usable_billing)

    async def test_metadata_permission_error_is_not_called_blocked(self):
        session = FakeSession([FakeResponse(403, {
            'code': 'permission_denied',
            'error': 'API key lacks permission',
        })])
        key = APIKey(Provider.XAI, 'test-key')

        result = await check_xai(key, session)

        self.assertTrue(result)
        self.assertTrue(key.permission_restricted)
        self.assertFalse(key.blocked)
        self.assertEqual(key.access_status, 'permission_restricted')

    async def test_rate_limited_key_is_not_marked_unusable(self):
        session = FakeSession(
            [key_metadata(), models_response(text_model('grok-4.3'))],
            [FakeResponse(
                429,
                {'code': 'rate_limit_error', 'error': 'Too many requests'},
                {'retry-after': '2'},
            )],
        )
        key = APIKey(Provider.XAI, 'test-key')

        result = await check_xai(key, session)

        self.assertTrue(result)
        self.assertTrue(key.rate_limited)
        self.assertIsNone(key.has_usable_billing)
        self.assertEqual(key.retry_after, '2')

    async def test_generic_bad_request_does_not_claim_usable_billing(self):
        session = FakeSession(
            [key_metadata(), models_response(text_model('grok-4.3'))],
            [FakeResponse(400, {
                'code': 'bad_request',
                'error': 'Bad data: unexpected field',
            })],
        )
        key = APIKey(Provider.XAI, 'test-key')

        result = await check_xai(key, session)

        self.assertTrue(result)
        self.assertIsNone(key.has_usable_billing)
        self.assertEqual(key.access_status, 'probe_error')

    async def test_model_error_falls_back(self):
        session = FakeSession(
            [
                key_metadata(),
                models_response(
                    text_model('grok-4.20-non-reasoning'),
                    text_model('grok-4.3'),
                ),
            ],
            [
                FakeResponse(404, {'error': 'Model not found'}),
                FakeResponse(200, {'usage': {'total_tokens': 12}}),
            ],
        )
        key = APIKey(Provider.XAI, 'test-key')

        result = await check_xai(key, session)

        self.assertTrue(result)
        self.assertIs(key.has_usable_billing, True)
        self.assertEqual(key.model, 'grok-4.3')

    async def test_api_outage_is_not_reported_as_invalid(self):
        session = FakeSession([FakeResponse(500, {'error': 'server error'})])
        key = APIKey(Provider.XAI, 'test-key')

        result = await check_xai(key, session)

        self.assertFalse(result)
        self.assertEqual(key.validation_status, 'unavailable')

    def test_model_order_prefers_non_reasoning_text_model(self):
        models = [
            text_model('grok-4.5', 20000, 60000),
            text_model('grok-4.20-reasoning'),
            text_model('grok-4.20-non-reasoning'),
            {'id': 'grok-imagine-video'},
        ]
        self.assertEqual(
            order_xai_text_models(models),
            [
                'grok-4.20-non-reasoning',
                'grok-4.5',
                'grok-4.20-reasoning',
            ],
        )


if __name__ == '__main__':
    unittest.main()
