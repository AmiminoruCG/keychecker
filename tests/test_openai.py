import asyncio
import ast
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import aiohttp
from APIKey import APIKey, Provider
import OpenAI as oai
from Azure import is_azure_key


class Response:
    def __init__(self, status=200, payload=None, headers=None, exception=None):
        self.status = status
        self.payload = payload if payload is not None else {}
        self.headers = headers or {}
        self.exception = exception

    async def __aenter__(self):
        if self.exception:
            raise self.exception
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self, **kwargs):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class Session:
    def __init__(self, gets=(), posts=()):
        self.gets, self.posts = list(gets), list(posts)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(('GET', url, kwargs))
        if not self.gets and url.endswith('/me'):
            return Response(404)
        return self.gets.pop(0)

    def post(self, url, **kwargs):
        self.calls.append(('POST', url, kwargs))
        return self.posts.pop(0)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def models(*ids):
    return Response(payload={'data': [{'id': model} for model in ids]})


def success(headers=None):
    return Response(payload={
        'choices': [{'message': {'role': 'assistant', 'content': 'OK'}, 'finish_reason': 'length'}],
        'usage': {'prompt_tokens': 10, 'completion_tokens': 1, 'total_tokens': 11},
    }, headers=headers)


def error(status, code, kind='invalid_request_error', message='fixture error', headers=None):
    return Response(status, {'error': {'code': code, 'type': kind, 'message': message}}, headers)


def new_key(secret='sk-proj-fixture-not-a-real-key'):
    return APIKey(Provider.OPENAI, secret)


class OpenAICheckerTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_proves_quota_with_small_real_request(self):
        session = Session([models('gpt-5-chat-latest', 'gpt-4o-mini')], [success({
            'x-ratelimit-limit-requests': '10000', 'x-ratelimit-limit-tokens': '10000000',
            'x-ratelimit-remaining-requests': '9999', 'x-ratelimit-remaining-tokens': '9999994',
        })])
        key = new_key()
        self.assertTrue(await oai.check_openai(key, session))
        self.assertTrue(oai.openai_key_usable(key))
        self.assertEqual(key.tier, 'Tier 4 (estimate)')
        self.assertEqual(key.usage_tokens, 11)
        self.assertEqual(key.tokens_remaining, 9999994)
        post = [call for call in session.calls if call[0] == 'POST'][0][2]
        self.assertEqual(post['json']['max_completion_tokens'], 1)
        self.assertFalse(post['json']['store'])
        self.assertFalse(post['allow_redirects'])
        self.assertTrue(post['json']['messages'][0]['content'])

    async def test_third_key_limits_match_nano_tier3_not_gpt5_tier4(self):
        session = Session([models('gpt-4.1-nano')], [success({
            'x-ratelimit-limit-requests': '5000', 'x-ratelimit-limit-tokens': '4000000',
        })])
        key = new_key()
        self.assertTrue(await oai.check_openai(key, session))
        self.assertEqual(key.tier, 'Tier 3 (estimate)')
        self.assertIn('4,000,000 TPM', oai.format_key_details(key))

    async def test_every_billing_code_is_terminal_and_valid_but_not_usable(self):
        for code in oai.BILLING_CODES:
            with self.subTest(code=code), patch.object(oai.asyncio, 'sleep', new_callable=AsyncMock) as sleep:
                key = new_key()
                session = Session([models('gpt-4o-mini')], [error(429, code)])
                self.assertTrue(await oai.check_openai(key, session))
                self.assertEqual(key.validation_status, 'valid')
                self.assertIs(key.has_quota, False)
                self.assertEqual(key.quota_status, code)
                self.assertFalse(oai.openai_key_usable(key))
                sleep.assert_not_awaited()
                self.assertEqual(sum(call[0] == 'POST' for call in session.calls), 1)

    async def test_insufficient_quota_type_without_code(self):
        key = new_key()
        session = Session([models('gpt-4o-mini')], [error(429, None, 'insufficient_quota')])
        self.assertTrue(await oai.check_openai(key, session))
        self.assertIs(key.has_quota, False)

    async def test_invalid_credential_returns_none(self):
        key = new_key()
        session = Session([error(401, 'invalid_api_key')])
        self.assertIsNone(await oai.check_openai(key, session))
        self.assertEqual(key.validation_status, 'invalid')
        self.assertEqual(len(session.calls), 1)

    async def test_key_revoked_between_listing_and_probe(self):
        key = new_key()
        session = Session([models('gpt-4o-mini')], [error(401, 'invalid_api_key')])
        self.assertIsNone(await oai.check_openai(key, session))
        self.assertEqual(key.validation_status, 'invalid')

    async def test_missing_or_malformed_rate_headers_do_not_break_success(self):
        for headers in ({}, {'x-ratelimit-limit-tokens': 'NaN', 'x-ratelimit-limit-requests': '-3'}):
            key = new_key()
            session = Session([models('gpt-4o-mini')], [success(headers)])
            self.assertTrue(await oai.check_openai(key, session))
            self.assertTrue(oai.openai_key_usable(key))
            self.assertIsNone(key.rpm)
            self.assertIsNone(key.tpm)
            self.assertEqual(key.tier, oai.unknown)

    async def test_generic_400_never_claims_quota_even_with_limit_headers(self):
        for headers in ({}, {'x-ratelimit-limit-tokens': '200000', 'x-ratelimit-limit-requests': '500'}):
            key = new_key()
            session = Session([models('gpt-4o-mini')], [error(400, 'unsupported_parameter', headers=headers)])
            self.assertFalse(await oai.check_openai(key, session))
            self.assertEqual(key.validation_status, 'valid')
            self.assertIsNone(key.has_quota)
            self.assertEqual(key.quota_status, 'probe_error')

    async def test_current_rate_limit_error_is_unknown_quota_not_exhausted(self):
        key = new_key()
        session = Session([models('gpt-4o-mini')], [error(429, 'slow_down', 'rate_limit_error', headers={'Retry-After': '120'})])
        with patch.object(oai.asyncio, 'sleep', new_callable=AsyncMock) as sleep:
            self.assertFalse(await oai.check_openai(key, session))
        self.assertEqual(key.validation_status, 'valid')
        self.assertIsNone(key.has_quota)
        self.assertTrue(key.rate_limited)
        self.assertEqual(key.retry_after, '120')
        self.assertFalse(oai.openai_key_usable(key))
        sleep.assert_not_awaited()

    async def test_retry_after_is_respected_then_success_clears_rate_status(self):
        key = new_key()
        session = Session([models('gpt-4o-mini')], [
            error(429, 'slow_down', 'rate_limit_error', headers={'Retry-After': '2'}), success(),
        ])
        with patch.object(oai.asyncio, 'sleep', new_callable=AsyncMock) as sleep:
            self.assertTrue(await oai.check_openai(key, session))
            sleep.assert_awaited_once_with(2)
        self.assertFalse(key.rate_limited)
        self.assertEqual(key.error_code, '')
        self.assertEqual(key.retry_after, '')

    async def test_server_errors_retry_at_most_three_times(self):
        key = new_key()
        session = Session([error(503, 'server_is_overloaded') for _ in range(3)])
        with patch.object(oai.asyncio, 'sleep', new_callable=AsyncMock) as sleep:
            self.assertFalse(await oai.check_openai(key, session, retries=10))
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(sleep.await_count, 2)
        self.assertEqual(key.validation_status, 'unknown')
        self.assertEqual(key.quota_status, 'api_error')

    async def test_server_error_on_probe_keeps_authentication_not_quota(self):
        key = new_key()
        session = Session([models('gpt-4o-mini')], [error(503, 'server_is_overloaded')])
        self.assertFalse(await oai.check_openai(key, session, retries=1))
        self.assertEqual(key.validation_status, 'valid')
        self.assertIsNone(key.has_quota)

    async def test_all_403_responses_do_not_prove_authentication(self):
        key = new_key()
        session = Session([error(403, 'forbidden')], [error(403, 'forbidden')])
        self.assertFalse(await oai.check_openai(key, session))
        self.assertEqual(key.validation_status, 'restricted')
        self.assertIsNone(key.has_quota)
        self.assertFalse(oai.openai_key_usable(key))

    async def test_model_listing_permission_can_differ_from_inference_permission(self):
        key = new_key()
        session = Session([error(403, 'insufficient_permissions')], [success()])
        self.assertTrue(await oai.check_openai(key, session))
        self.assertFalse(key.access_to_model_listing)
        self.assertFalse(key.permission_restricted)
        self.assertTrue(oai.openai_key_usable(key))

    async def test_401_missing_scope_is_restricted_not_invalid(self):
        key = new_key()
        session = Session([error(401, None, message='Missing scopes: api.model.read')], [success()])
        self.assertTrue(await oai.check_openai(key, session))
        self.assertTrue(oai.openai_key_usable(key))

    async def test_country_and_ip_blocks_are_not_bad_credentials(self):
        for status, code in ((403, 'unsupported_country_region_territory'), (401, 'ip_not_authorized')):
            key = new_key()
            session = Session([error(status, code)])
            self.assertFalse(await oai.check_openai(key, session))
            self.assertEqual(key.validation_status, 'restricted')
            self.assertTrue(key.blocked)
            self.assertEqual(len(session.calls), 1)

    async def test_retired_model_is_skipped_before_any_probe(self):
        key = new_key()
        session = Session([models('gpt-5-chat-latest', 'gpt-4.1-nano')], [success()])
        self.assertTrue(await oai.check_openai(key, session))
        self.assertEqual(key.model, 'gpt-4.1-nano')

    async def test_missing_model_falls_back_then_reports_exhausted_credit(self):
        key = new_key()
        session = Session([models('gpt-4.1-nano', 'gpt-4o-mini')], [
            error(404, 'model_not_found'), error(429, 'credit_balance_exhausted', 'insufficient_quota'),
        ])
        self.assertTrue(await oai.check_openai(key, session))
        self.assertEqual(key.model, 'gpt-4o-mini')
        self.assertIs(key.has_quota, False)

    async def test_denied_model_falls_back_to_permitted_model(self):
        key = new_key()
        session = Session([models('gpt-4.1-nano', 'gpt-4o-mini')], [error(403, 'model_not_allowed'), success()])
        self.assertTrue(await oai.check_openai(key, session))
        self.assertTrue(oai.openai_key_usable(key))
        self.assertEqual(key.model, 'gpt-4o-mini')

    async def test_only_specialized_models_does_not_mean_invalid_or_no_balance(self):
        key = new_key()
        session = Session([models('gpt-image-1', 'text-embedding-3-small')])
        self.assertFalse(await oai.check_openai(key, session))
        self.assertEqual(key.validation_status, 'valid')
        self.assertIsNone(key.has_quota)
        self.assertEqual(key.quota_status, 'no_suitable_model')
        self.assertFalse(any(call[0] == 'POST' for call in session.calls))

    async def test_new_text_model_can_be_probed(self):
        key = new_key()
        session = Session([models('gpt-6-luna')], [success()])
        self.assertTrue(await oai.check_openai(key, session))
        self.assertEqual(key.model, 'gpt-6-luna')

    async def test_response_only_fallback_is_bounded_and_accepts_truncated_completion(self):
        key = new_key()
        session = Session([models('gpt-6-luna')], [
            error(400, 'unsupported_model', message='Use the Responses API for this model'),
            Response(payload={'object': 'response', 'status': 'incomplete', 'output': [],
                              'usage': {'input_tokens': 10, 'output_tokens': 16, 'total_tokens': 26}}),
        ])
        self.assertTrue(await oai.check_openai(key, session))
        posts = [call for call in session.calls if call[0] == 'POST']
        self.assertTrue(posts[1][1].endswith('/responses'))
        self.assertEqual(posts[1][2]['json']['max_output_tokens'], 16)
        self.assertEqual(key.usage_tokens, 26)

    async def test_malformed_json_and_200_html_are_not_success(self):
        for payload in (ValueError('not json'), {}, [], {'data': 'bad'}):
            key = new_key()
            self.assertFalse(await oai.check_openai(key, Session([Response(payload=payload)])))
            self.assertEqual(key.quota_status, 'invalid_response')
            self.assertIsNone(key.has_quota)
        key = new_key()
        self.assertFalse(await oai.check_openai(key, Session([models('gpt-4o-mini')], [Response(payload={})])))
        self.assertIsNone(key.has_quota)

    async def test_malformed_error_fields_are_inconclusive_not_a_crash(self):
        key = new_key()
        session = Session([models('gpt-4o-mini')], [
            Response(400, {'error': {'code': [], 'type': {}, 'message': None}}),
        ])
        self.assertFalse(await oai.check_openai(key, session))
        self.assertIsNone(key.has_quota)

    async def test_model_fallback_attempts_are_bounded(self):
        key = new_key()
        session = Session([models('gpt-4.1-nano', 'gpt-4o-mini', 'gpt-5-nano', 'gpt-4o')],
                          [error(404, 'model_not_found') for _ in range(3)])
        self.assertFalse(await oai.check_openai(key, session))
        self.assertEqual(sum(call[0] == 'POST' for call in session.calls), 3)
        self.assertEqual(key.validation_status, 'valid')
        self.assertIsNone(key.has_quota)

    async def test_redirects_are_not_followed_or_interpreted_as_invalid(self):
        key = new_key()
        session = Session([Response(302, headers={'location': 'https://example.invalid/'})])
        self.assertFalse(await oai.check_openai(key, session))
        self.assertIsNone(key.has_quota)
        self.assertFalse(session.calls[0][2]['allow_redirects'])
        self.assertEqual(len(session.calls), 1)

    async def test_optional_org_metadata_failure_preserves_usable_key(self):
        for response in (Response(403), Response(404), Response(200, {'orgs': 'bad'}),
                         Response(exception=asyncio.TimeoutError())):
            key = new_key()
            session = Session([models('gpt-4o-mini'), response], [success()])
            self.assertTrue(await oai.check_openai(key, session))
            self.assertTrue(oai.openai_key_usable(key))
            self.assertEqual(key.org_metadata_status, 'unavailable')

    async def test_org_ids_not_names_are_used_and_clones_have_fresh_state(self):
        key = new_key('legacy-fixture')
        key.default_org = 'org-main'
        key.model = 'gpt-4o'
        key.has_quota = True
        session = Session([Response(payload={'orgs': {'data': [
            {'id': 'org-main', 'name': 'Main name', 'is_default': True},
            {'id': 'org-other', 'name': 'Other name'},
        ]}})])
        self.assertTrue(await oai.get_oai_org(key, session))
        self.assertEqual(key.organizations, ['org-main', 'org-other'])
        clone_session = Session([models('gpt-4.1-nano')], [success()])
        clones = await oai.clone_key(key, clone_session)
        self.assertEqual(len(clones), 1)
        clone = next(iter(clones))
        self.assertEqual(clone.model, 'gpt-4.1-nano')
        self.assertEqual(key.model, 'gpt-4o')
        for _, _, kwargs in clone_session.calls:
            self.assertEqual(kwargs['headers']['OpenAI-Organization'], 'org-other')
        self.assertEqual(clone_session.calls[1][2]['json']['model'], 'gpt-4.1-nano')
        self.assertIsNot(key.organization_names, clone.organization_names)

    async def test_failed_clone_never_inherits_usable_quota(self):
        key = new_key('legacy-fixture')
        key.default_org = 'org-main'
        key.organizations = ['org-main', 'org-other']
        key.has_quota = True
        key.validation_status = 'valid'
        session = Session([models('gpt-4o-mini')], [error(403, 'forbidden')])
        self.assertFalse(await oai.clone_key(key, session))

    async def test_scoped_keys_never_try_other_organizations(self):
        for prefix in ('sk-proj-', 'sk-svcacct-'):
            key = new_key(prefix + 'fixture')
            key.default_org = 'org-main'
            key.organizations = ['org-main', 'org-other']
            session = Session()
            self.assertFalse(await oai.clone_key(key, session))
            self.assertEqual(session.calls, [])

    async def test_verifyorg_success_is_access_not_identity_and_uses_same_org(self):
        key = new_key()
        key.has_quota = True
        key.rpm = 5000
        session = Session(posts=[success()])
        self.assertIsNone(await oai.check_id_verified(key, session, org='org-target'))
        self.assertEqual(key.org_verification_status, 'o3_access_confirmed')
        self.assertEqual(session.calls[0][2]['headers']['OpenAI-Organization'], 'org-target')
        self.assertIs(key.has_quota, True)
        self.assertEqual(key.rpm, 5000)

    async def test_verifyorg_explicit_requirement_only_and_no_hot_loop(self):
        key = new_key()
        session = Session(posts=[error(403, 'forbidden', message='Your organization must be verified')])
        self.assertIs(await oai.check_id_verified(key, session), False)
        self.assertEqual(key.org_verification_status, 'verification_required')
        for response in (error(400, 'unsupported_parameter'),
                         error(429, 'credit_balance_exhausted', 'insufficient_quota'),
                         error(429, 'slow_down', headers={'retry-after': '60'})):
            key = new_key()
            with patch.object(oai.asyncio, 'sleep', new_callable=AsyncMock) as sleep:
                session = Session(posts=[response])
                self.assertIsNone(await oai.check_id_verified(key, session))
                self.assertEqual(len(session.calls), 1)
                sleep.assert_not_awaited()

    async def test_network_get_retries_but_ambiguous_post_does_not(self):
        key = new_key()
        session = Session([Response(exception=aiohttp.ClientConnectionError()), models('gpt-4o-mini')],
                          [Response(exception=asyncio.TimeoutError())])
        with patch.object(oai.asyncio, 'sleep', new_callable=AsyncMock) as sleep:
            self.assertFalse(await oai.check_openai(key, session))
        self.assertEqual(sleep.await_count, 1)
        self.assertEqual(sum(call[0] == 'POST' for call in session.calls), 1)
        self.assertEqual(key.validation_status, 'valid')
        self.assertIsNone(key.has_quota)
        self.assertEqual(key.quota_status, 'network_error')

    async def test_new_run_clears_previous_success(self):
        key = new_key()
        self.assertTrue(await oai.check_openai(key, Session([models('gpt-4o-mini')], [success()])))
        self.assertFalse(await oai.check_openai(key, Session([error(503, 'server_is_overloaded')]), retries=1))
        self.assertIsNone(key.has_quota)
        self.assertFalse(oai.openai_key_usable(key))
        self.assertEqual(key.validation_status, 'unknown')

    async def test_error_messages_redact_credentials(self):
        key = new_key()
        session = Session([models('gpt-4o-mini')], [error(400, 'bad', message='Bad key: ' + key.api_key)])
        await oai.check_openai(key, session)
        self.assertNotIn(key.api_key, key.error_message)
        self.assertIn('[REDACTED]', key.error_message)

    async def test_tier_is_model_specific_and_requires_both_headers(self):
        self.assertEqual(await oai.get_oai_key_tier(2000000, 'gpt-4o', 10000), 'Tier 4 (estimate)')
        self.assertEqual(await oai.get_oai_key_tier(2000000, 'gpt-5', 5000), 'Tier 3 (estimate)')
        self.assertEqual(await oai.get_oai_key_tier(4000000, 'gpt-4.1-nano-2025-04-14', 5000), 'Tier 3 (estimate)')
        self.assertEqual(await oai.get_oai_key_tier(2000000, 'gpt-4o'), oai.unknown)
        self.assertEqual(await oai.get_oai_key_tier(123456, 'gpt-4o', 10000), oai.unknown)
        self.assertEqual(await oai.get_oai_key_tier(4000000, 'gpt-6-luna', 5000), oai.unknown)

    def test_order_is_deterministic_and_ignores_retired_or_specialized_models(self):
        ids = ['gpt-4o', 'gpt-4.1-nano', 'gpt-5-chat-latest', 'gpt-4o-audio-preview', 'o3-deep-research']
        self.assertEqual(oai.order_oai_models(ids), ['gpt-4.1-nano', 'gpt-4o'])
        self.assertEqual(oai.order_oai_models(ids), oai.order_oai_models(reversed(ids)))

    def test_key_routing_supports_scoped_keys_without_internal_marker(self):
        for prefix in ('sk-proj-', 'sk-svcacct-'):
            self.assertTrue(oai.is_openai_key(prefix + 'A_b-' * 10))
        self.assertTrue(oai.is_openai_key('sk-' + 'a' * 20 + 'T3BlbkFJ' + 'b' * 20))
        for key in ('sk-ant-' + 'a' * 90, 'sk-or-v1-' + 'a' * 64, 'sk-' + 'a' * 32, 'sk-proj-short'):
            self.assertFalse(oai.is_openai_key(key))

    def test_retry_after_http_date_and_invalid_values(self):
        date = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=120))
        self.assertIsNone(oai.retry_delay({'retry-after': date}, 0))
        for value in ('NaN', 'Infinity', 'not a date', '-1'):
            self.assertGreaterEqual(oai.retry_delay({'retry-after': value}, 0), 0.5)
        self.assertEqual(oai.retry_delay({'retry-after': '3'}, 0), 3)

    def test_unknown_keys_have_separate_group_and_never_export(self):
        key = new_key()
        key.quota_status = 'api_error'
        key.has_quota = True  # defensive: stale boolean alone is not enough
        self.assertFalse(oai.openai_key_usable(key))
        key.has_quota = None
        output = io.StringIO()
        with redirect_stdout(output):
            oai.pretty_print_oai_keys([key])
        self.assertIn('Unconfirmed or restricted', output.getvalue())
        self.assertIn('0 authenticated', output.getvalue())
        self.assertNotIn('Valid keys with billing/quota exhausted', output.getvalue())


class OpenAIMainIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def load_functions(self, *names, **scope):
        source = ast.parse(Path('main.py').read_text(encoding='utf-8'))
        nodes = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'main.py', 'exec'), scope)
        return scope

    async def test_validate_openai_preserves_inconclusive_key(self):
        key = new_key()
        scope = self.load_functions(
            'validate_openai', APIKey=APIKey,
            aiohttp=SimpleNamespace(ClientTimeout=aiohttp.ClientTimeout, ClientSession=lambda **kw: Session()),
            IO=SimpleNamespace(conditional_print=lambda *args: None),
            args=SimpleNamespace(verbose=False, verifyorg=False), api_keys=set(), cloned_keys=set(),
            check_openai=AsyncMock(return_value=False), clone_key=AsyncMock(return_value=set()),
            openai_status_text=oai.openai_status_text,
        )
        await scope['validate_openai'](key, asyncio.Semaphore(1))
        self.assertIn(key, scope['api_keys'])
        scope['clone_key'].assert_not_awaited()

    async def test_main_routes_legacy_and_scoped_keys_only_to_openai(self):
        legacy = 'sk-' + 'a' * 20 + 'T3BlbkFJ' + 'b' * 20
        inputs = {'sk-proj-' + 'z' * 80, 'sk-svcacct-' + 'y' * 80, legacy}
        validator = AsyncMock()
        scope = self.load_functions(
            'validate_keys', asyncio=asyncio, inputted_keys=inputs, APIKey=APIKey, Provider=Provider,
            is_openai_key=oai.is_openai_key, validate_openai=validator, openai_semaphore=asyncio.Semaphore(1),
            is_azure_key=is_azure_key,
            api_keys=set(), as_completed=lambda futures: (),
        )
        await scope['validate_keys']()
        self.assertEqual(validator.await_count, 3)
        self.assertTrue(all(call.args[0].provider == Provider.OPENAI for call in validator.await_args_list))

    def test_proxy_output_contains_only_confirmed_usable_keys_without_duplicates(self):
        good = new_key('fixture-usable')
        good.validation_status, good.has_quota = 'valid', True
        same_key_other_org = new_key('fixture-usable')
        same_key_other_org.validation_status, same_key_other_org.has_quota = 'valid', True
        exhausted = new_key('fixture-exhausted')
        exhausted.validation_status, exhausted.has_quota = 'valid', False
        unknown = new_key('fixture-unknown')
        unknown.has_quota = True  # old state must not bypass validation_status
        limited = new_key('fixture-rate-limited')
        limited.validation_status, limited.has_quota, limited.rate_limited = 'valid', True, True
        scope = self.load_functions(
            'output_keys', args=SimpleNamespace(nooutput=True, proxyoutput=True),
            asyncio=SimpleNamespace(run=lambda result: None), validate_keys=lambda: None,
            api_keys={good, same_key_other_org, exhausted, unknown, limited}, Provider=Provider,
            openai_key_usable=oai.openai_key_usable,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            scope['output_keys']()
        line = next(line for line in output.getvalue().splitlines() if line.startswith('OPENAI_KEY='))
        self.assertEqual(line, 'OPENAI_KEY=fixture-usable')


if __name__ == '__main__':
    unittest.main()
