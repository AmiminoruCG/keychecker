import asyncio
import ast
from contextlib import redirect_stdout
import hashlib
import io
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import requests

from APIKey import APIKey, Provider
import Azure as azure


SECRET = 'a' * 32
ENDPOINT = 'https://test-resource.openai.azure.com'


class Response:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self.payload = {} if payload is None else payload
        self.headers = {} if headers is None else headers

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class Session:
    """Strict in-memory HTTP transport: unexpected requests fail the test."""

    def __init__(self, gets=(), posts=()):
        self.gets = list(gets)
        self.posts = list(posts)
        self.calls = []

    def request(self, method, url, **kwargs):
        method = method.upper()
        self.calls.append((method, url, kwargs))
        queue = self.gets if method == 'GET' else self.posts
        if not queue:
            raise AssertionError(f'Unexpected {method} request to fixture endpoint')
        response = queue.pop(0)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(method, url, kwargs)
        return response

    def close(self):
        pass


def deployment(identifier='chat', model='gpt-4o', status='succeeded'):
    return {'id': identifier, 'model': model, 'status': status}


def deployments(*items):
    return Response(payload={'data': list(items)})


def success():
    return Response(payload={
        'id': 'chatcmpl-fixture',
        'object': 'chat.completion',
        'model': 'gpt-4o',
        'choices': [{
            'index': 0,
            'message': {'role': 'assistant', 'content': 'OK'},
            'finish_reason': 'stop',
        }],
        'usage': {'prompt_tokens': 9, 'completion_tokens': 1, 'total_tokens': 10},
    })


def error(status, code, message='fixture error', headers=None):
    return Response(status, {'error': {'code': code, 'message': message}}, headers)


def new_key(value=None):
    return APIKey(Provider.AZURE, value or f'test-resource:{SECRET}')


def posts(session):
    return [call for call in session.calls if call[0] == 'POST']


class AzureInputTests(unittest.TestCase):
    def test_resource_and_endpoint_formats_normalize_identically(self):
        inputs = (
            f'test-resource:{SECRET}',
            f'{ENDPOINT}:{SECRET}',
            f'{ENDPOINT}/:{SECRET}',
            f'{ENDPOINT}/openai/v1/:{SECRET}',
        )
        for value in inputs:
            with self.subTest(value=value):
                self.assertEqual(azure.parse_azure_key(value), (ENDPOINT, SECRET, ''))
                self.assertTrue(azure.is_azure_key(value))

    def test_deployment_url_preserves_real_deployment_name(self):
        value = f'{ENDPOINT}/openai/deployments/Production_Chat/chat/completions?api-version=2024-10-21:{SECRET}'
        self.assertEqual(azure.parse_azure_key(value), (ENDPOINT, SECRET, 'Production_Chat'))

    def test_unsafe_hosts_and_malformed_inputs_are_rejected_before_http(self):
        inputs = (
            f'http://test-resource.openai.azure.com:{SECRET}',
            f'https://example.com:{SECRET}',
            f'https://test-resource.openai.azure.com.example.com:{SECRET}',
            f'https://user:password@test-resource.openai.azure.com:{SECRET}',
            f'https://test-resource.openai.azure.com:444:{SECRET}',
            f'test-resource.openai.azure.com/path:{SECRET}',
            'test-resource:',
            ':empty-resource',
            'no-colon',
        )
        for value in inputs:
            with self.subTest(value=value):
                self.assertFalse(azure.is_azure_key(value))
                with self.assertRaises(ValueError):
                    azure.parse_azure_key(value)


class AzureCheckerTests(unittest.TestCase):
    def setUp(self):
        sleep_patch = patch('time.sleep')
        self.sleep = sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def test_success_requires_inference_and_records_usage_without_filter_claim(self):
        key = new_key()
        session = Session([deployments(deployment())], [success()])
        self.assertIs(azure.check_azure(key, session=session), True)
        self.assertEqual(key.validation_status, 'valid')
        self.assertIs(key.has_quota, True)
        self.assertEqual(key.endpoint, ENDPOINT)
        self.assertEqual(key.resource_name, 'test-resource')
        self.assertEqual(key.best_deployment, 'chat')
        self.assertEqual(key.model, 'gpt-4o')
        self.assertEqual(key.usage_tokens, 10)
        self.assertEqual(key.input_tokens_used, 9)
        self.assertEqual(key.output_tokens_used, 1)
        self.assertIsNot(key.unfiltered, True)
        self.assertTrue(azure.azure_key_usable(key))
        self.assertTrue(any(report['id'] == 'chat' and report['status'] == 'usable'
                            for report in key.deployments))
        for _, _, kwargs in session.calls:
            self.assertIsNotNone(kwargs.get('timeout'))
            self.assertIs(kwargs.get('allow_redirects'), False)

    def test_gpt51_uses_deployment_id_and_compatible_small_token_cap(self):
        key = new_key()
        session = Session([deployments(deployment('GPT_4o', 'gpt-5.1'))], [success()])
        self.assertTrue(azure.check_azure(key, session=session))
        _, url, kwargs = posts(session)[0]
        self.assertEqual(url, f'{ENDPOINT}/openai/v1/chat/completions')
        body = kwargs['json']
        self.assertEqual(body['model'], 'GPT_4o')
        self.assertEqual(body['max_completion_tokens'], 16)
        self.assertNotIn('max_tokens', body)
        self.assertIs(body['store'], False)
        self.assertEqual(key.model, 'gpt-5.1')

    def test_reasoning_model_is_checked(self):
        key = new_key()
        session = Session([deployments(deployment('reasoning', 'o3-mini'))], [success()])
        self.assertTrue(azure.check_azure(key, session=session))
        self.assertEqual(posts(session)[0][2]['json']['model'], 'reasoning')

    def test_responses_only_model_uses_responses_endpoint(self):
        key = new_key()
        response = Response(payload={
            'id': 'resp_fixture', 'object': 'response', 'status': 'completed',
            'output': [{'id': 'msg_fixture', 'type': 'message', 'role': 'assistant',
                        'status': 'completed',
                        'content': [{'type': 'output_text', 'text': 'OK', 'annotations': []}]}],
            'usage': {'input_tokens': 9, 'output_tokens': 1, 'total_tokens': 10},
        })
        session = Session([deployments(deployment('pro-deployment', 'gpt-5-pro'))], [response])
        self.assertTrue(azure.check_azure(key, session=session))
        _, url, kwargs = posts(session)[0]
        self.assertEqual(url, f'{ENDPOINT}/openai/v1/responses')
        self.assertEqual(kwargs['json']['model'], 'pro-deployment')
        self.assertEqual(kwargs['json']['max_output_tokens'], 16)
        self.assertIs(kwargs['json']['store'], False)
        self.assertTrue(azure.azure_key_usable(key))
        self.assertEqual(azure.azure_proxy_credential(key), '')

    def test_failed_and_specialized_deployments_are_retained_but_not_called(self):
        items = [
            deployment('broken', 'gpt-4o', 'failed'),
            deployment('pictures', 'dall-e-3'),
            deployment('audio', 'gpt-4o-audio-preview'),
            deployment('realtime', 'gpt-4o-realtime-preview'),
            deployment('embeddings', 'text-embedding-3-large'),
            deployment('chat', 'gpt-4o-mini'),
        ]
        key = new_key()
        session = Session([deployments(*items)], [success()])
        self.assertTrue(azure.check_azure(key, session=session))
        self.assertEqual(len(posts(session)), 1)
        self.assertEqual(posts(session)[0][2]['json']['model'], 'chat')
        reports = {report['id']: report for report in key.deployments}
        self.assertEqual(set(reports), {item['id'] for item in items})
        self.assertEqual(reports['broken']['status'], 'failed')
        for identifier in ('pictures', 'audio', 'realtime', 'embeddings'):
            self.assertNotEqual(reports[identifier]['status'], 'usable')

    def test_only_specialized_deployments_do_not_count_as_usable(self):
        key = new_key()
        session = Session([deployments(deployment('image', 'dall-e-3'))])
        self.assertIs(azure.check_azure(key, session=session), False)
        self.assertEqual(key.validation_status, 'valid')
        self.assertIsNone(key.has_quota)
        self.assertEqual(key.best_deployment, '')
        self.assertFalse(azure.azure_key_usable(key))
        self.assertFalse(posts(session))

    def test_empty_deployment_list_authenticates_but_does_not_confirm_quota(self):
        key = new_key()
        session = Session([deployments()])
        self.assertIs(azure.check_azure(key, session=session), False)
        self.assertEqual(key.validation_status, 'valid')
        self.assertIsNone(key.has_quota)
        self.assertEqual(key.best_deployment, '')

    def test_discovery_401_is_invalid_and_never_runs_inference(self):
        key = new_key()
        session = Session([error(401, 'Unauthorized')])
        self.assertIsNone(azure.check_azure(key, session=session))
        self.assertEqual(key.validation_status, 'invalid')
        self.assertFalse(azure.azure_key_usable(key))
        self.assertEqual(len(session.calls), 1)

    def test_discovery_restrictions_or_transient_errors_are_not_invalid_credentials(self):
        for status, code in ((403, 'Forbidden'), (429, 'RateLimitReached'),
                             (503, 'ServiceUnavailable')):
            with self.subTest(status=status):
                key = new_key()
                session = Session([error(status, code) for _ in range(6)])
                self.assertIs(azure.check_azure(key, session=session), False)
                self.assertNotEqual(key.validation_status, 'invalid')
                self.assertIsNone(key.has_quota)
                self.assertFalse(azure.azure_key_usable(key))
                self.assertFalse(posts(session))

    def test_failed_inference_never_becomes_a_successful_key(self):
        cases = ((401, 'Unauthorized'), (403, 'Forbidden'),
                 (429, 'RateLimitReached'), (503, 'ServiceUnavailable'))
        for status, code in cases:
            with self.subTest(status=status):
                key = new_key()
                session = Session([deployments(deployment())],
                                  [error(status, code) for _ in range(3)])
                self.assertIs(azure.check_azure(key, session=session), False)
                self.assertEqual(key.validation_status, 'valid')
                self.assertIsNone(key.has_quota)
                self.assertEqual(key.best_deployment, '')
                self.assertFalse(azure.azure_key_usable(key))
                self.assertTrue(key.deployments)
                self.assertTrue(all(report['status'] != 'usable' for report in key.deployments))
                if status in (401, 403):
                    self.assertEqual(len(posts(session)), 1)

    def test_billing_error_is_not_retried_or_classified_as_invalid(self):
        key = new_key()
        session = Session([deployments(deployment())],
                          [error(429, 'insufficient_quota', 'Quota has been exhausted')])
        self.assertIs(azure.check_azure(key, session=session), False)
        self.assertEqual(key.validation_status, 'valid')
        self.assertIs(key.has_quota, False)
        self.assertEqual(len(posts(session)), 1)
        self.assertFalse(azure.azure_key_usable(key))

    def test_malformed_success_does_not_confirm_usable_inference(self):
        for payload in ({}, {'choices': []}, {'choices': [{'message': {'content': 'OK'}}]},
                        {'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}},
                        ValueError('invalid JSON')):
            with self.subTest(payload=repr(payload)):
                key = new_key()
                session = Session([deployments(deployment())], [Response(payload=payload)])
                self.assertIs(azure.check_azure(key, session=session), False)
                self.assertIsNone(key.has_quota)
                self.assertIsNot(key.unfiltered, True)
                self.assertFalse(azure.azure_key_usable(key))

    def test_malformed_message_or_usage_cannot_pass_as_chat_success(self):
        cases = []
        for message in ({}, {'role': 'user', 'content': 'OK'}):
            payload = success().payload
            payload['choices'][0]['message'] = message
            cases.append(payload)
        for usage in ([], None, {'prompt_tokens': -1, 'completion_tokens': 1},
                      {'prompt_tokens': True, 'completion_tokens': 1}):
            payload = success().payload
            payload['usage'] = usage
            cases.append(payload)
        for payload in cases:
            with self.subTest(payload=payload):
                key = new_key()
                session = Session([deployments(deployment())], [Response(payload=payload)])
                self.assertIs(azure.check_azure(key, session=session), False)
                self.assertIsNone(key.has_quota)
                self.assertFalse(azure.azure_key_usable(key))

    def test_completed_response_with_empty_output_is_not_usable(self):
        key = new_key()
        payload = {'object': 'response', 'status': 'completed', 'output': [],
                   'usage': {'input_tokens': 9, 'output_tokens': 1, 'total_tokens': 10}}
        session = Session([deployments(deployment('pro', 'gpt-5-pro'))], [Response(payload=payload)])
        self.assertIs(azure.check_azure(key, session=session), False)
        self.assertIsNone(key.has_quota)
        self.assertFalse(azure.azure_key_usable(key))

    def test_malformed_error_payload_is_retained_without_raising(self):
        for payload in ([], None, {'error': 'not-an-object'}, {'error': []},
                        {'error': {'code': [], 'message': {'nested': 'error'}}}):
            with self.subTest(payload=payload):
                key = new_key()
                session = Session([deployments(deployment())], [Response(400, payload=payload)])
                self.assertIs(azure.check_azure(key, session=session), False)
                self.assertEqual(key.validation_status, 'valid')
                self.assertIsNone(key.has_quota)
                self.assertFalse(azure.azure_key_usable(key))
                with redirect_stdout(io.StringIO()):
                    azure.pretty_print_azure_keys([key])

    def test_malformed_discovery_is_visible_and_does_not_raise(self):
        for payload in ({}, {'data': None}, {'data': {}}, ValueError('invalid JSON')):
            with self.subTest(payload=repr(payload)):
                key = new_key()
                # A supported fallback may authenticate with a model catalogue.
                session = Session([Response(payload=payload), deployments()])
                self.assertIs(azure.check_azure(key, session=session), False)
                self.assertNotEqual(key.validation_status, 'invalid')
                self.assertFalse(azure.azure_key_usable(key))
                self.assertIsNone(key.has_quota)
                self.assertFalse(posts(session))

    def test_malformed_deployment_entries_do_not_hide_valid_entries(self):
        key = new_key()
        session = Session([deployments(None, {}, 'invalid-entry', {'model': 'gpt-4o'}, deployment())],
                          [success()])
        self.assertTrue(azure.check_azure(key, session=session))
        self.assertEqual(key.best_deployment, 'chat')
        self.assertEqual(len(posts(session)), 1)

    def test_same_resource_pagination_is_followed_and_deployments_deduplicated(self):
        key = new_key()
        first = Response(payload={
            'data': [deployment('first')],
            'nextLink': f'{ENDPOINT}/openai/deployments?api-version=2022-12-01&after=first',
        })
        session = Session([first, deployments(deployment('first'), deployment('second'))],
                          [success(), success()])
        self.assertTrue(azure.check_azure(key, session=session))
        self.assertTrue(key.discovery_complete)
        self.assertEqual({item['id'] for item in key.deployments}, {'first', 'second'})
        self.assertEqual(len(posts(session)), 2)

    def test_pagination_cannot_forward_secret_to_another_host(self):
        key = new_key()
        first = Response(payload={
            'data': [deployment()],
            'nextLink': 'https://example.com/openai/deployments?api-version=2022-12-01',
        })
        session = Session([first], [success()])
        self.assertTrue(azure.check_azure(key, session=session))
        self.assertFalse(key.discovery_complete)
        self.assertEqual(key.discovery_status, 'partial')
        self.assertTrue(all(call[1].startswith(ENDPOINT + '/') for call in session.calls))
        self.assertEqual(len([call for call in session.calls if call[0] == 'GET']), 1)

    def test_failed_later_page_keeps_discovered_deployments_and_partial_status(self):
        key = new_key()
        first = Response(payload={
            'data': [deployment()],
            'nextLink': f'{ENDPOINT}/openai/deployments?api-version=2022-12-01&after=chat',
        })
        session = Session([first] + [error(503, 'ServiceUnavailable') for _ in range(3)],
                          [success()])
        self.assertTrue(azure.check_azure(key, session=session))
        self.assertEqual(key.validation_status, 'valid')
        self.assertEqual(key.best_deployment, 'chat')
        self.assertFalse(key.discovery_complete)
        self.assertEqual(key.discovery_status, 'partial')

    def test_malformed_earlier_page_keeps_discovery_partial_after_last_page(self):
        key = new_key()
        first = Response(payload={
            'data': [None, deployment('first')],
            'nextLink': f'{ENDPOINT}/openai/deployments?api-version=2022-12-01&after=first',
        })
        session = Session([first, deployments(deployment('second'))], [success(), success()])
        self.assertTrue(azure.check_azure(key, session=session))
        self.assertFalse(key.discovery_complete)
        self.assertEqual(key.discovery_status, 'partial')

    def test_later_page_401_does_not_erase_earlier_authentication_proof(self):
        key = new_key()
        first = Response(payload={
            'data': [deployment()],
            'nextLink': f'{ENDPOINT}/openai/deployments?api-version=2022-12-01&after=chat',
        })
        session = Session([first, error(401, 'Unauthorized')], [error(403, 'Forbidden')])
        self.assertIs(azure.check_azure(key, session=session), False)
        self.assertEqual(key.validation_status, 'valid')
        self.assertFalse(key.discovery_complete)
        self.assertEqual(key.discovery_status, 'partial')
        self.assertEqual(len(key.deployments), 1)
        self.assertFalse(azure.azure_key_usable(key))

    def test_inference_probe_budget_keeps_unchecked_deployments_visible(self):
        key = new_key()
        items = [deployment(f'chat-{index:02}') for index in range(25)]
        session = Session([deployments(*items)], [success() for _ in range(20)])
        self.assertTrue(azure.check_azure(key, session=session))
        self.assertEqual(len(posts(session)), 20)
        self.assertEqual(len(key.deployments), 25)
        self.assertEqual(sum(item['status'] == 'usable' for item in key.deployments), 20)
        self.assertEqual(sum(item['status'] == 'not_checked' for item in key.deployments), 5)

    def test_quota_failure_does_not_cover_deployments_left_outside_probe_budget(self):
        key = new_key()
        session = Session([deployments(deployment('first'), deployment('second'))],
                          [error(429, 'insufficient_quota')])
        with patch.object(azure, 'MAX_PROBES', 1):
            self.assertIs(azure.check_azure(key, session=session), False)
        self.assertEqual(key.validation_status, 'valid')
        self.assertIsNone(key.has_quota)
        self.assertEqual(len(posts(session)), 1)
        self.assertEqual(sum(item['status'] == 'quota_exhausted' for item in key.deployments), 1)
        self.assertEqual(sum(item['status'] == 'not_checked' for item in key.deployments), 1)

    def test_deadline_after_discovery_marks_eligible_deployment_not_checked(self):
        key = new_key()
        session = Session([deployments(deployment())])
        with patch.object(azure.time, 'monotonic', side_effect=lambda: 100 if session.calls else 0):
            self.assertIs(azure.check_azure(key, session=session), False)
        self.assertEqual(key.validation_status, 'valid')
        self.assertEqual(key.access_status, 'not_checked')
        self.assertIsNone(key.has_quota)
        self.assertEqual(key.deployments[0]['status'], 'not_checked')
        self.assertFalse(posts(session))

    def test_expired_check_budget_does_not_start_requests(self):
        key = new_key()
        session = Session()
        with patch.object(azure, 'CHECK_SECONDS', 0):
            self.assertIs(azure.check_azure(key, session=session), False)
        self.assertEqual(session.calls, [])
        self.assertIsNone(key.has_quota)
        self.assertFalse(azure.azure_key_usable(key))

    def test_rechecking_failed_key_clears_previous_usable_state(self):
        key = new_key()
        self.assertTrue(azure.check_azure(key, session=Session([deployments(deployment())], [success()])))
        self.assertTrue(azure.azure_key_usable(key))
        self.assertIsNone(azure.check_azure(key, session=Session([error(401, 'Unauthorized')])))
        self.assertEqual(key.validation_status, 'invalid')
        self.assertIsNone(key.has_quota)
        self.assertEqual(key.best_deployment, '')
        self.assertFalse(azure.azure_key_usable(key))
        self.assertEqual(azure.azure_proxy_credential(key), '')

    def test_catalogue_fallback_cannot_invent_deployment_names(self):
        key = new_key()
        session = Session([
            error(404, 'NotFound'),
            Response(payload={'data': [{'id': 'gpt-4o'}, {'id': 'gpt-5.1'}]}),
        ])
        self.assertIs(azure.check_azure(key, session=session), False)
        self.assertEqual(key.validation_status, 'valid')
        self.assertFalse(key.discovery_complete)
        self.assertIsNone(key.has_quota)
        self.assertEqual(key.best_deployment, '')
        self.assertFalse(posts(session))
        self.assertTrue(any('/openai/v1/models' in call[1] for call in session.calls))

    def test_explicit_deployment_url_can_probe_without_legacy_discovery(self):
        key = new_key(f'{ENDPOINT}/openai/deployments/Production_Chat/chat/completions?api-version=2024-10-21:{SECRET}')
        session = Session([error(404, 'NotFound'), deployments()], [success()])
        self.assertTrue(azure.check_azure(key, session=session))
        self.assertEqual(key.best_deployment, 'Production_Chat')
        self.assertEqual(posts(session)[0][2]['json']['model'], 'Production_Chat')

    def test_transient_server_error_is_retried_up_to_a_success(self):
        key = new_key()
        session = Session([error(503, 'ServiceUnavailable'), deployments(deployment())], [success()])
        self.assertTrue(azure.check_azure(key, session=session))
        self.assertEqual(len([call for call in session.calls if call[0] == 'GET']), 2)
        self.assertTrue(self.sleep.called)

    def test_transient_inference_retries_are_bounded(self):
        key = new_key()
        session = Session([deployments(deployment())],
                          [error(503, 'ServiceUnavailable') for _ in range(3)])
        self.assertIs(azure.check_azure(key, session=session, retries=3), False)
        self.assertEqual(len(posts(session)), 3)
        self.assertIsNone(key.has_quota)

    def test_long_retry_after_defers_inference_without_immediate_retry(self):
        key = new_key()
        session = Session([deployments(deployment())],
                          [error(429, 'RateLimitReached', headers={'Retry-After': '120'})])
        self.assertIs(azure.check_azure(key, session=session), False)
        self.assertEqual(len(posts(session)), 1)
        self.assertIsNone(key.has_quota)
        self.sleep.assert_not_called()

    def test_inference_timeout_is_not_replayed(self):
        key = new_key()
        session = Session([deployments(deployment())], [requests.ReadTimeout('fixture timeout')])
        self.assertIs(azure.check_azure(key, session=session), False)
        self.assertEqual(len(posts(session)), 1)
        self.assertEqual(key.validation_status, 'valid')
        self.assertIsNone(key.has_quota)
        self.assertFalse(azure.azure_key_usable(key))

    def test_best_deployment_is_selected_only_among_successful_ones(self):
        key = new_key()
        def outcome(method, url, kwargs):
            return error(403, 'Forbidden') if kwargs['json']['model'] == 'top' else success()
        session = Session([deployments(deployment('top', 'gpt-5.1'),
                                       deployment('working', 'gpt-4o-mini'))],
                          [outcome, outcome])
        self.assertTrue(azure.check_azure(key, session=session))
        self.assertEqual(key.best_deployment, 'working')
        self.assertEqual(key.model, 'gpt-4o-mini')

    def test_best_model_prefers_full_model_over_mini_and_gpt5_over_gpt4(self):
        for items, expected in (([deployment('mini', 'gpt-4o-mini'), deployment('full', 'gpt-4o')], 'full'),
                                ([deployment('old', 'gpt-4o'), deployment('new', 'gpt-5.1')], 'new')):
            with self.subTest(expected=expected):
                key = new_key()
                session = Session([deployments(*items)], [success(), success()])
                self.assertTrue(azure.check_azure(key, session=session))
                self.assertEqual(key.best_deployment, expected)

    def test_output_never_labels_missing_annotations_as_unfiltered(self):
        key = new_key()
        session = Session([deployments(deployment())], [success()])
        azure.check_azure(key, session=session)
        output = io.StringIO()
        with redirect_stdout(output):
            azure.pretty_print_azure_keys([key])
        self.assertNotIn('!!!UNFILTERED!!!', output.getvalue())

    def test_proxy_format_uses_resource_and_confirmed_deployment(self):
        key = new_key(f'{ENDPOINT}/:{SECRET}')
        session = Session([deployments(deployment('Production_Chat'))], [success()])
        azure.check_azure(key, session=session)
        self.assertEqual(azure.azure_proxy_credential(key), f'test-resource:Production_Chat:{SECRET}')
        key.access_status = 'unknown'
        self.assertFalse(azure.azure_key_usable(key))
        self.assertEqual(azure.azure_proxy_credential(key), '')


class AzureMainIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def load_functions(self, *names, **scope):
        source = ast.parse(Path('main.py').read_text(encoding='utf-8'))
        nodes = [node for node in source.body
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'main.py', 'exec'), scope)
        return scope

    async def test_wrapper_preserves_inconclusive_result(self):
        key = new_key()
        checker = Mock(return_value=False)
        scope = self.load_functions(
            'validate_azure', APIKey=APIKey, asyncio=asyncio, hashlib=hashlib,
            check_azure=checker, api_keys=set(), azure_status_text=azure.azure_status_text,
            IO=SimpleNamespace(conditional_print=lambda *args: None), args=SimpleNamespace(verbose=False),
        )
        await scope['validate_azure'](key, asyncio.Semaphore(1))
        self.assertIn(key, scope['api_keys'])
        self.assertIsNone(key.has_quota)
        checker.assert_called_once_with(key)

    async def test_wrapper_reports_worker_exception_without_leaking_secret(self):
        key = new_key()
        scope = self.load_functions(
            'validate_azure', APIKey=APIKey, asyncio=asyncio, hashlib=hashlib,
            check_azure=Mock(side_effect=RuntimeError(f'request failed with {key.api_key}')),
            api_keys=set(), azure_status_text=azure.azure_status_text,
            IO=SimpleNamespace(conditional_print=lambda *args: None), args=SimpleNamespace(verbose=False),
        )
        output = io.StringIO()
        with redirect_stdout(output):
            await scope['validate_azure'](key, asyncio.Semaphore(1))
        self.assertIn(key, scope['api_keys'])
        self.assertEqual(key.validation_status, 'unknown')
        self.assertIsNone(key.has_quota)
        self.assertIn('RuntimeError', output.getvalue())
        self.assertNotIn(SECRET, output.getvalue())

    async def test_main_routes_azure_resource_containing_sk_before_generic_sk_keys(self):
        inputs = {f'test-resource:{SECRET}', f'ask-service:{SECRET}', f'{ENDPOINT}/:{SECRET}'}
        validator = AsyncMock()
        scope = self.load_functions(
            'validate_keys', asyncio=asyncio, inputted_keys=inputs, APIKey=APIKey, Provider=Provider,
            is_azure_key=azure.is_azure_key, validate_azure=validator, azure_semaphore=asyncio.Semaphore(1),
            api_keys=set(), as_completed=lambda futures: (),
        )
        await scope['validate_keys']()
        self.assertEqual(validator.await_count, len(inputs))
        self.assertTrue(all(call.args[0].provider == Provider.AZURE for call in validator.await_args_list))

    async def test_main_still_routes_aws_credentials_to_aws_validator(self):
        value = 'AKIA' + '0' * 16 + ':' + 'b' * 40
        self.assertFalse(azure.is_azure_key(value))
        source = ast.parse(Path('main.py').read_text(encoding='utf-8'))
        regex_nodes = [node for node in source.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'aws_regex' for target in node.targets)]
        regex_scope = {'re': re}
        exec(compile(ast.Module(body=regex_nodes, type_ignores=[]), 'main.py', 'exec'), regex_scope)
        aws_validator, azure_validator = AsyncMock(), AsyncMock()
        async def retry_wrapper(function, key, semaphore, attempts):
            return await function(key, semaphore)
        scope = self.load_functions(
            'validate_keys', asyncio=asyncio, inputted_keys={value}, APIKey=APIKey, Provider=Provider,
            is_azure_key=azure.is_azure_key, validate_azure=azure_validator,
            is_openai_key=lambda key: False, moonshot_regex=SimpleNamespace(fullmatch=lambda key: None),
            aws_regex=regex_scope['aws_regex'], args=SimpleNamespace(awslegacy=False),
            execute_with_retries=retry_wrapper, validate_aws_async=aws_validator,
            concurrent_connections=asyncio.Semaphore(1), api_keys=set(), as_completed=lambda futures: (),
        )
        await scope['validate_keys']()
        azure_validator.assert_not_awaited()
        self.assertEqual(aws_validator.await_count, 1)
        self.assertEqual(aws_validator.await_args.args[0].provider, Provider.AWS)

    def test_proxy_output_excludes_inconclusive_keys(self):
        good, unknown = new_key(), new_key('unconfirmed:' + 'b' * 32)
        azure.check_azure(good, session=Session([deployments(deployment('verified'))], [success()]))
        # A stale quota boolean alone must not permit export.
        unknown.has_quota = True
        scope = self.load_functions(
            'output_keys', args=SimpleNamespace(nooutput=True, proxyoutput=True),
            asyncio=SimpleNamespace(run=lambda result: None), validate_keys=lambda: None,
            api_keys={good, unknown}, Provider=Provider,
            openai_key_usable=lambda key: False,
            azure_key_usable=azure.azure_key_usable, azure_proxy_credential=azure.azure_proxy_credential,
        )
        output = io.StringIO()
        with redirect_stdout(output):
            scope['output_keys']()
        line = next(line for line in output.getvalue().splitlines() if line.startswith('AZURE_CREDENTIALS='))
        self.assertEqual(line, f'AZURE_CREDENTIALS=test-resource:verified:{SECRET}')


if __name__ == '__main__':
    unittest.main()
