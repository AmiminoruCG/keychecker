import asyncio
import ast
import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from APIKey import APIKey, Provider
import OpenRouter


class FakeResponse:
    def __init__(self, status=200, payload=None, *, enter_error=None):
        self.status = status
        self.payload = payload
        self.enter_error = enter_error
        self.headers = {}

    async def __aenter__(self):
        if self.enter_error is not None:
            raise self.enter_error
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def json(self, **kwargs):
        if isinstance(self.payload, BaseException):
            raise self.payload
        return self.payload


class FakeSession:
    """Ordered fake requests: no credentials or network calls leave the tests."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError(f'Unexpected request: {url}')
        return self.responses.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False


def make_key():
    return APIKey(Provider.OPENROUTER, 'test-openrouter-key')


def key_response(**overrides):
    data = {
        'usage': 17,
        'usage_daily': 1,
        'usage_weekly': 4,
        'usage_monthly': 9,
        'limit': 20,
        'limit_remaining': 3,
        'limit_reset': 'monthly',
        'is_free_tier': False,
    }
    data.update(overrides)
    return FakeResponse(payload={'data': data})


def credits_response(**overrides):
    data = {'total_credits': 100, 'total_usage': 25}
    data.update(overrides)
    return FakeResponse(payload={'data': data})


def output_line(output, label):
    for line in output.splitlines():
        if label.lower() in line.lower():
            return line
    raise AssertionError(f'Missing output label {label!r}:\n{output}')


class OpenRouterCheckerTests(unittest.IsolatedAsyncioTestCase):
    async def test_account_balance_and_key_budget_are_distinct(self):
        key = make_key()
        session = FakeSession(key_response(), credits_response())

        self.assertIs(await OpenRouter.check_openrouter(key, session), True)

        self.assertEqual(key.validation_status, 'valid')
        self.assertEqual(key.account_balance, 75)
        self.assertEqual(key.balance, 75)
        self.assertEqual(key.total_credits, 100)
        self.assertEqual(key.total_usage, 25)
        self.assertEqual(key.limit_remaining, 3)
        self.assertEqual(OpenRouter.get_openrouter_spending_budget(key), 3)
        self.assertIs(key.credits_api_available, True)
        self.assertEqual(key.credits_api_status, 'ok')
        self.assertEqual(
            [url for url, _ in session.calls],
            [
                'https://openrouter.ai/api/v1/key',
                'https://openrouter.ai/api/v1/credits',
            ],
        )

    async def test_numeric_strings_are_accepted(self):
        key = make_key()
        session = FakeSession(
            key_response(usage='17', limit='20', limit_remaining='3'),
            credits_response(total_credits='100.00', total_usage='25'),
        )

        self.assertTrue(await OpenRouter.check_openrouter(key, session))

        self.assertEqual(key.usage, 17)
        self.assertEqual(key.credit_limit, 20)
        self.assertEqual(key.account_balance, 75)
        self.assertEqual(OpenRouter.get_openrouter_spending_budget(key), 3)

    async def test_explicit_zero_credits_is_known_zero(self):
        key = make_key()
        session = FakeSession(
            key_response(), credits_response(total_credits=0, total_usage=0)
        )

        self.assertTrue(await OpenRouter.check_openrouter(key, session))

        self.assertEqual(key.account_balance, 0)
        self.assertEqual(key.balance, 0)
        self.assertIs(key.credits_api_available, True)
        self.assertEqual(key.credits_api_status, 'ok')
        self.assertEqual(OpenRouter.get_openrouter_spending_budget(key), 0)

    async def test_credits_http_failure_does_not_invalidate_authenticated_key(self):
        cases = [
            (403, 'permission_denied'),
            (429, 'rate_limited'),
            (500, 'http_error'),
            (503, 'http_error'),
        ]
        for status, expected in cases:
            with self.subTest(status=status):
                key = make_key()
                session = FakeSession(key_response(), FakeResponse(status, {}))

                self.assertIs(await OpenRouter.check_openrouter(key, session), True)

                self.assertEqual(key.validation_status, 'valid')
                self.assertIsNone(key.account_balance)
                self.assertIsNone(key.balance)
                self.assertIs(key.credits_api_available, False)
                self.assertEqual(key.credits_api_status, expected)
                self.assertIsNone(OpenRouter.get_openrouter_spending_budget(key))

    async def test_credits_timeout_does_not_invalidate_authenticated_key(self):
        key = make_key()
        session = FakeSession(
            key_response(), FakeResponse(enter_error=asyncio.TimeoutError())
        )

        self.assertIs(await OpenRouter.check_openrouter(key, session), True)

        self.assertEqual(key.validation_status, 'valid')
        self.assertEqual(key.credits_api_status, 'network_error')
        self.assertIsNone(key.account_balance)
        self.assertIsNone(key.balance)
        self.assertIs(key.credits_api_available, False)

    async def test_malformed_credits_response_is_unknown_instead_of_zero(self):
        payloads = [
            ValueError('Invalid JSON'),
            None,
            [],
            {},
            {'data': None},
            {'data': []},
            {'data': {}},
            {'data': {'total_credits': 100}},
            {'data': {'total_usage': 25}},
        ]
        for payload in payloads:
            with self.subTest(payload=repr(payload)):
                key = make_key()
                session = FakeSession(key_response(), FakeResponse(payload=payload))

                self.assertIs(await OpenRouter.check_openrouter(key, session), True)

                self.assertEqual(key.validation_status, 'valid')
                self.assertEqual(key.credits_api_status, 'invalid_response')
                self.assertIsNone(key.account_balance)
                self.assertIsNone(key.balance)
                self.assertIs(key.credits_api_available, False)

    async def test_invalid_credit_numbers_are_never_known_balances(self):
        for field in ('total_credits', 'total_usage'):
            for invalid in (None, True, False, 'invalid', '', [], {}, float('nan'),
                            float('inf'), float('-inf'), 'NaN', 'Infinity'):
                with self.subTest(field=field, value=repr(invalid)):
                    key = make_key()
                    session = FakeSession(
                        key_response(), credits_response(**{field: invalid})
                    )

                    self.assertIs(await OpenRouter.check_openrouter(key, session), True)

                    self.assertEqual(key.validation_status, 'valid')
                    self.assertEqual(key.credits_api_status, 'invalid_response')
                    self.assertIsNone(key.account_balance)
                    self.assertIsNone(key.balance)
                    self.assertIs(key.credits_api_available, False)

    async def test_authentication_rejection_is_invalid_and_skips_credits(self):
        for status in (401, 403):
            with self.subTest(status=status):
                key = make_key()
                session = FakeSession(FakeResponse(status, {}))

                self.assertIsNone(await OpenRouter.check_openrouter(key, session))

                self.assertEqual(key.validation_status, 'invalid')
                self.assertEqual(key.credits_api_status, 'not_checked')
                self.assertEqual(len(session.calls), 1)

    async def test_key_endpoint_outage_is_preserved_as_inconclusive(self):
        for status in (429, 500, 503):
            with self.subTest(status=status):
                key = make_key()
                session = FakeSession(FakeResponse(status, {}))

                self.assertIs(await OpenRouter.check_openrouter(key, session), True)

                self.assertEqual(key.validation_status, 'unknown')
                self.assertEqual(key.credits_api_status, 'not_checked')
                self.assertEqual(len(session.calls), 1)
                self.assertIsNone(OpenRouter.get_openrouter_spending_budget(key))

    async def test_key_endpoint_timeout_is_preserved_as_inconclusive(self):
        key = make_key()
        session = FakeSession(FakeResponse(enter_error=asyncio.TimeoutError()))

        self.assertIs(await OpenRouter.check_openrouter(key, session), True)

        self.assertEqual(key.validation_status, 'unknown')
        self.assertEqual(key.credits_api_status, 'not_checked')
        self.assertIsNone(key.account_balance)

    async def test_malformed_key_metadata_is_preserved_as_inconclusive(self):
        for payload in (ValueError('Invalid JSON'), None, [], {}, {'data': None},
                        {'data': []}, {'data': 'not metadata'}):
            with self.subTest(payload=repr(payload)):
                key = make_key()
                session = FakeSession(FakeResponse(payload=payload))

                self.assertIs(await OpenRouter.check_openrouter(key, session), True)

                self.assertEqual(key.validation_status, 'unknown')
                self.assertEqual(key.credits_api_status, 'not_checked')
                self.assertEqual(len(session.calls), 1)

    async def test_empty_metadata_dictionary_is_inconclusive(self):
        key = make_key()
        session = FakeSession(FakeResponse(payload={'data': {}}))

        self.assertTrue(await OpenRouter.check_openrouter(key, session))

        self.assertEqual(key.validation_status, 'unknown')
        self.assertEqual(key.credits_api_status, 'not_checked')
        self.assertEqual(len(session.calls), 1)
        self.assertIs(key.key_limit_known, False)
        self.assertIsNone(key.usage)
        self.assertIsNone(OpenRouter.get_openrouter_spending_budget(key))

    async def test_rechecking_same_key_does_not_reuse_stale_balance(self):
        for second_status in ('invalid', 'unknown', 'credits_unavailable'):
            with self.subTest(second_status=second_status):
                key = make_key()
                self.assertIs(await OpenRouter.check_openrouter(
                    key, FakeSession(key_response(), credits_response())
                ), True)
                self.assertEqual(key.account_balance, 75)
                self.assertEqual(OpenRouter.get_openrouter_spending_budget(key), 3)
                if second_status == 'invalid':
                    session = FakeSession(FakeResponse(401, {}))
                elif second_status == 'unknown':
                    session = FakeSession(FakeResponse(enter_error=asyncio.TimeoutError()))
                else:
                    session = FakeSession(key_response(), FakeResponse(403, {}))

                result = await OpenRouter.check_openrouter(key, session)

                self.assertIs(result, None if second_status == 'invalid' else True)
                self.assertEqual(
                    key.validation_status,
                    'valid' if second_status == 'credits_unavailable' else second_status,
                )
                self.assertIsNone(key.account_balance)
                self.assertIsNone(key.balance)
                self.assertIsNone(key.total_credits)
                self.assertIsNone(key.total_usage)
                self.assertIs(key.credits_api_available, False)
                self.assertIsNone(OpenRouter.get_openrouter_spending_budget(key))
                if second_status != 'credits_unavailable':
                    self.assertIs(key.key_limit_known, False)
                    self.assertIsNone(key.limit_remaining)
                    self.assertIsNone(key.usage)

    async def test_cancellation_propagates_from_each_endpoint(self):
        for endpoint in ('key', 'credits'):
            for stage in ('request', 'json'):
                with self.subTest(endpoint=endpoint, stage=stage):
                    response = (
                        FakeResponse(enter_error=asyncio.CancelledError())
                        if stage == 'request'
                        else FakeResponse(payload=asyncio.CancelledError())
                    )
                    responses = [response] if endpoint == 'key' else [key_response(), response]
                    session = FakeSession(*responses)

                    with self.assertRaises(asyncio.CancelledError):
                        await OpenRouter.check_openrouter(make_key(), session)


class OpenRouterBalanceTests(unittest.TestCase):
    def parsed_key(self, data, account_balance=75, validation_status='valid'):
        key = make_key()
        OpenRouter.parse_openrouter_key_info(key, data)
        key.validation_status = validation_status
        key.account_balance = account_balance
        return key

    def test_missing_numeric_metadata_is_unknown(self):
        key = self.parsed_key({})

        for field in ('usage', 'usage_daily', 'usage_weekly', 'usage_monthly',
                      'credit_limit', 'limit_remaining'):
            with self.subTest(field=field):
                self.assertIsNone(getattr(key, field))
        self.assertIs(key.key_limit_known, False)

    def test_bad_numeric_metadata_is_unknown_without_crashing(self):
        for invalid in (True, False, 'invalid', '', [], {}, float('nan'),
                        float('inf'), float('-inf'), 'NaN', 'Infinity'):
            with self.subTest(value=repr(invalid)):
                key = self.parsed_key({
                    'usage': invalid,
                    'usage_daily': invalid,
                    'usage_weekly': invalid,
                    'usage_monthly': invalid,
                    'limit': invalid,
                    'limit_remaining': invalid,
                })

                for field in ('usage', 'usage_daily', 'usage_weekly', 'usage_monthly',
                              'credit_limit', 'limit_remaining'):
                    self.assertIsNone(getattr(key, field), field)
                self.assertIs(key.key_limit_known, False)
                self.assertIsNone(OpenRouter.get_openrouter_spending_budget(key))

    def test_explicit_unlimited_limit_uses_available_account_balance(self):
        key = self.parsed_key({'limit': None})

        self.assertIs(key.key_limit_known, True)
        self.assertIsNone(key.credit_limit)
        self.assertEqual(OpenRouter.get_openrouter_spending_budget(key), 75)
        limit_line = output_line(
            OpenRouter.format_openrouter_key(key), 'Key spending limit'
        ).lower()
        self.assertTrue('unlimited' in limit_line or 'no per-key cap' in limit_line)

    def test_unknown_limit_is_not_treated_as_unlimited(self):
        key = self.parsed_key({})

        self.assertIsNone(OpenRouter.get_openrouter_spending_budget(key))
        line = output_line(OpenRouter.format_openrouter_key(key), 'Key spending limit')
        self.assertIn('unknown', line.lower())
        self.assertNotIn('unlimited', line.lower())

    def test_finite_limit_without_remaining_has_unknown_budget(self):
        key = self.parsed_key({'limit': 20, 'limit_remaining': None})

        self.assertIs(key.key_limit_known, True)
        self.assertIsNone(OpenRouter.get_openrouter_spending_budget(key))
        line = output_line(OpenRouter.format_openrouter_key(key), 'Remaining key limit')
        self.assertIn('unknown', line.lower())
        self.assertNotIn('unlimited', line.lower())

    def test_zero_key_limit_prevents_spending_with_positive_account_balance(self):
        key = self.parsed_key({'limit': 0, 'limit_remaining': 0})

        self.assertEqual(key.account_balance, 75)
        self.assertEqual(OpenRouter.get_openrouter_spending_budget(key), 0)
        self.assertIs(key.limit_reached, True)

    def test_budget_is_bounded_by_both_balance_and_remaining_limit(self):
        for balance, remaining, expected in (
            (75, 3, 3), (2, 3, 2), (0, 3, 0), (-2, 3, 0), (75, 0, 0), (75, -2, 0),
        ):
            with self.subTest(balance=balance, remaining=remaining):
                key = self.parsed_key({'limit': 20, 'limit_remaining': remaining}, balance)
                self.assertEqual(OpenRouter.get_openrouter_spending_budget(key), expected)

    def test_unknown_input_does_not_hide_known_exhaustion(self):
        cases = [
            ({}, 0),
            ({}, -1),
            ({'limit': 20, 'limit_remaining': 0}, None),
            ({'limit_remaining': 0}, 75),
            ({'limit_remaining': -1}, 75),
        ]
        for metadata, balance in cases:
            with self.subTest(metadata=metadata, balance=balance):
                key = self.parsed_key(metadata, balance)
                self.assertEqual(OpenRouter.get_openrouter_spending_budget(key), 0)

    def test_explicit_no_cap_ignores_inconsistent_remaining_limit(self):
        for remaining in (0, -1):
            with self.subTest(remaining=remaining):
                key = self.parsed_key({'limit': None, 'limit_remaining': remaining})

                self.assertIs(key.limit_reached, False)
                self.assertEqual(OpenRouter.get_openrouter_spending_budget(key), 75)

    def test_remaining_key_limit_alone_is_not_money_in_the_account(self):
        key = self.parsed_key({'limit': 20, 'limit_remaining': 3}, None)
        key.credits_api_status = 'permission_denied'
        key.credits_api_http_status = 403
        output = OpenRouter.format_openrouter_key(key)

        self.assertIsNone(OpenRouter.get_openrouter_spending_budget(key))
        for label in ('Account balance', 'Budget from balance/key limit'):
            line = output_line(output, label)
            self.assertIn('unknown', line.lower())
            self.assertNotIn('$', line)
        self.assertIn('management key', output.lower())
        self.assertIn('403', output)

    def test_unknown_authentication_prevents_a_budget_claim(self):
        key = self.parsed_key(
            {'limit': 20, 'limit_remaining': 3}, validation_status='unknown'
        )

        self.assertIsNone(OpenRouter.get_openrouter_spending_budget(key))

    def test_money_none_means_unknown(self):
        self.assertEqual(OpenRouter.format_openrouter_money(None), 'unknown')
        self.assertEqual(OpenRouter.format_openrouter_money(0), '$0.0000')

    def test_usage_windows_and_limit_reset_remain_independent(self):
        key = self.parsed_key({
            'usage': 17,
            'usage_daily': 1,
            'usage_weekly': 4,
            'usage_monthly': 9,
            'limit': 20,
            'limit_remaining': 3,
            'limit_reset': 'monthly',
        })

        self.assertEqual(
            (key.usage, key.usage_daily, key.usage_weekly, key.usage_monthly),
            (17, 1, 4, 9),
        )
        self.assertEqual(key.limit_reset, 'monthly')
        self.assertEqual(OpenRouter.get_openrouter_spending_budget(key), 3)
        output = OpenRouter.format_openrouter_key(key)
        for amount in ('$17.0000', '$1.0000', '$4.0000', '$9.0000'):
            self.assertIn(amount, output)
        self.assertIn('monthly', output.lower())

    def test_output_separates_balance_limit_and_budget_in_english(self):
        key = self.parsed_key({'limit': 20, 'limit_remaining': 3})
        output = OpenRouter.format_openrouter_key(key)

        self.assertGreater(len(output.splitlines()), 3)
        for label, amount in (
            ('Account balance', '$75.0000'),
            ('Key spending limit', '$20.0000'),
            ('Remaining key limit', '$3.0000'),
            ('Budget from balance/key limit', '$3.0000'),
        ):
            self.assertIn(amount, output_line(output, label))

    def test_deprecated_rate_limit_is_not_displayed_as_rpm(self):
        key = self.parsed_key({
            'limit': None,
            'rate_limit': {'requests': 200, 'interval': '10s'},
        })
        output = io.StringIO()
        with redirect_stdout(output):
            OpenRouter.pretty_print_openrouter_keys([key])

        self.assertNotIn('RPM', output.getvalue().upper())


class OpenRouterMainIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def load_validator(self, session):
        """Load the real pipeline wrapper without importing main's CLI startup."""
        source_path = Path(__file__).resolve().parents[1] / 'main.py'
        source = ast.parse(source_path.read_text(encoding='utf-8'))
        nodes = [
            node for node in source.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == 'validate_openrouter'
        ]
        self.assertEqual(len(nodes), 1)
        scope = {
            'APIKey': APIKey,
            'aiohttp': SimpleNamespace(ClientSession=lambda: session),
            'IO': SimpleNamespace(conditional_print=Mock()),
            'args': SimpleNamespace(verbose=True),
            'api_keys': set(),
            'check_openrouter': AsyncMock(wraps=OpenRouter.check_openrouter),
            'openrouter_status_text': OpenRouter.openrouter_status_text,
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path), 'exec'), scope)
        return scope

    def console_text(self, scope):
        return '\n'.join(call.args[0] for call in scope['IO'].conditional_print.call_args_list)

    async def test_main_retains_inconclusive_key_without_calling_it_valid(self):
        key = make_key()
        session = FakeSession(FakeResponse(429, {}))
        scope = self.load_validator(session)

        await scope['validate_openrouter'](key, asyncio.Semaphore(1))

        scope['check_openrouter'].assert_awaited_once_with(key, session)
        self.assertEqual(key.validation_status, 'unknown')
        self.assertIn(key, scope['api_keys'])
        output = self.console_text(scope).lower()
        self.assertIn('unknown', output)
        self.assertNotIn('is valid', output)
        self.assertNotIn('authenticated', output)
        self.assertNotIn('invalid openrouter key', output)

    async def test_main_drops_invalid_key(self):
        key = make_key()
        session = FakeSession(FakeResponse(401, {}))
        scope = self.load_validator(session)

        await scope['validate_openrouter'](key, asyncio.Semaphore(1))

        scope['check_openrouter'].assert_awaited_once_with(key, session)
        self.assertEqual(key.validation_status, 'invalid')
        self.assertNotIn(key, scope['api_keys'])
        self.assertIn('invalid openrouter key', self.console_text(scope).lower())

    async def test_main_keeps_authenticated_key_when_credits_require_management_key(self):
        key = make_key()
        session = FakeSession(key_response(), FakeResponse(403, {}))
        scope = self.load_validator(session)

        await scope['validate_openrouter'](key, asyncio.Semaphore(1))

        scope['check_openrouter'].assert_awaited_once_with(key, session)
        self.assertEqual(key.validation_status, 'valid')
        self.assertIn(key, scope['api_keys'])
        self.assertIsNone(key.account_balance)
        self.assertEqual(key.credits_api_status, 'permission_denied')
        output = self.console_text(scope).lower()
        self.assertIn('authenticated', output)
        self.assertNotIn('invalid openrouter key', output)


if __name__ == '__main__':
    unittest.main()
