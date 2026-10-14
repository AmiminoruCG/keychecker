"""Offline behavioral tests for the Linux launcher (no provider credentials)."""

import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile


LAUNCHER = Path(__file__).resolve().parents[1] / 'run.sh'
FIXTURE_NAME = 'keychecker-runtime-fixture'


def write_runtime_wheel(directory, version):
    """Build one local wheel that supplies the launcher's import smoke check."""
    files = {
        f'{name}/__init__.py': ''
        for name in ('aiohttp', 'boto3', 'botocore', 'requests', 'certifi', 'google',
                     'google/oauth2', 'google/auth', 'google/auth/transport')
    }
    files['google/oauth2/service_account.py'] = '# Offline test fixture.\n'
    files['google/auth/transport/requests.py'] = 'class Request:\n    pass\n'
    dist_info = f'keychecker_runtime_fixture-{version}.dist-info'
    files[f'{dist_info}/METADATA'] = (
        'Metadata-Version: 2.1\n'
        f'Name: {FIXTURE_NAME}\nVersion: {version}\nRequires-Python: >=3.9\n'
    )
    files[f'{dist_info}/WHEEL'] = (
        'Wheel-Version: 1.0\nGenerator: offline-test\n'
        'Root-Is-Purelib: true\nTag: py3-none-any\n'
    )
    record = io.StringIO()
    writer = csv.writer(record, lineterminator='\n')
    for name, content in files.items():
        encoded = content.encode('utf-8')
        digest = base64.urlsafe_b64encode(hashlib.sha256(encoded).digest()).rstrip(b'=').decode()
        writer.writerow((name, f'sha256={digest}', len(encoded)))
    writer.writerow((f'{dist_info}/RECORD', '', ''))
    files[f'{dist_info}/RECORD'] = record.getvalue()
    wheel = directory / f'keychecker_runtime_fixture-{version}-py3-none-any.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, content in files.items():
            archive.writestr(name, content)


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'Linux/POSIX Bash launcher')
class RunShTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='keychecker launcher ')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repository with spaces'
        self.caller = self.root / 'caller with spaces'
        self.wheels = self.root / 'offline wheelhouse'
        self.hooks = self.root / 'python hooks'
        for directory in (self.repo, self.caller, self.wheels, self.hooks):
            directory.mkdir()
        shutil.copyfile(LAUNCHER, self.repo / 'run.sh')
        self.requirements = self.repo / 'requirements.txt'
        self.requirements.write_text(f'{FIXTURE_NAME}==1.0\n', encoding='utf-8')
        self.venv = self.repo / '.venv'
        self.marker = self.root / 'child-started'
        self.pip_log = self.root / 'pip-calls.jsonl'
        write_runtime_wheel(self.wheels, '1.0')
        write_runtime_wheel(self.wheels, '2.0')
        (self.repo / 'main.py').write_text(textwrap.dedent('''\
            import importlib.metadata
            import json
            import os
            from pathlib import Path
            import sys

            Path(os.environ['TEST_CHILD_MARKER']).write_text('started')
            arguments = sys.argv[1:]
            result = {
                'argv': arguments,
                'cwd': os.getcwd(),
                'stdin': sys.stdin.read(),
                'prefix': sys.prefix,
                'fixture_version': importlib.metadata.version('keychecker-runtime-fixture'),
            }
            if '-file' in arguments:
                result['file_content'] = Path(arguments[arguments.index('-file') + 1]).read_text()
            print(json.dumps(result))
            raise SystemExit(int(os.environ.get('TEST_CHILD_EXIT', '0')))
            '''), encoding='utf-8')
        # Observe real `python -m pip` calls without replacing pip or its resolver.
        (self.hooks / 'sitecustomize.py').write_text(textwrap.dedent('''\
            import atexit
            import json
            import os
            import sys

            def record_pip(arguments):
                with open(os.environ['TEST_PIP_LOG'], 'a', encoding='utf-8') as stream:
                    stream.write(json.dumps(arguments) + '\\n')

            original = getattr(sys, 'orig_argv', [])
            for index in range(len(original) - 1):
                if original[index:index + 2] == ['-m', 'pip']:
                    record_pip(original[index + 2:])
                    break
            else:
                # Python 3.9 has no orig_argv. runpy sets argv[0] before exit.
                def record_legacy_pip():
                    if sys.argv and sys.argv[0].replace('\\\\', '/').endswith('/pip/__main__.py'):
                        record_pip(sys.argv[1:])
                atexit.register(record_legacy_pip)
            '''), encoding='utf-8')
        self.env = os.environ.copy()
        self.env.pop('KEYCHECKER_VENV', None)
        self.env.update({
            'KEYCHECKER_PYTHON': sys.executable,
            'PIP_CONFIG_FILE': os.devnull,
            'PIP_NO_INDEX': '1',
            'PIP_FIND_LINKS': self.wheels.as_uri(),
            'PIP_DISABLE_PIP_VERSION_CHECK': '1',
            'PIP_NO_CACHE_DIR': '1',
            'PYTHONPATH': str(self.hooks),
            'PYTHONNOUSERSITE': '1',
            'TEST_CHILD_MARKER': str(self.marker),
            'TEST_PIP_LOG': str(self.pip_log),
        })

    def launch(self, *arguments, stdin='', overrides=None):
        env = self.env.copy()
        env.update(overrides or {})
        return subprocess.run(
            ['bash', str(self.repo / 'run.sh'), *arguments],
            cwd=self.caller, env=env, input=stdin, text=True,
            capture_output=True, timeout=120,
        )

    def assert_child(self, result, *, status=0, venv=None):
        self.assertEqual(result.returncode, status, result.stderr)
        self.assertTrue(self.marker.exists(), result.stderr)
        # Any setup output on stdout makes this fail: stdout belongs to main.py.
        output = json.loads(result.stdout)
        self.assertEqual(Path(output['prefix']), venv or self.venv)
        return output

    def assert_setup_failed(self, result):
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.marker.exists(), result.stdout)
        self.assertEqual(result.stdout, '')
        self.assertTrue(result.stderr)

    def pip_calls(self):
        if not self.pip_log.exists():
            return []
        return [json.loads(line) for line in self.pip_log.read_text().splitlines()]

    def install_calls(self):
        return [call for call in self.pip_calls() if 'install' in call and '--dry-run' not in call]

    def reset_observations(self):
        self.marker.unlink(missing_ok=True)
        self.pip_log.unlink(missing_ok=True)

    def isolated_commands(self):
        commands = self.root / 'isolated commands'
        commands.mkdir()
        for name in ('bash', 'dirname'):
            executable = shutil.which(name)
            self.assertIsNotNone(executable)
            (commands / name).symlink_to(executable)
        self.env.pop('KEYCHECKER_PYTHON', None)
        self.env.update({
            'PATH': str(commands),
            'TEST_BASE_PYTHON': getattr(sys, '_base_executable', sys.executable),
        })
        return commands

    def fake_command(self, commands, name, body):
        script = commands / name
        script.write_text('#!' + shutil.which('bash') + '\n' + textwrap.dedent(body), encoding='utf-8')
        script.chmod(0o755)

    def test_first_install_then_healthy_environment_stays_offline_without_install(self):
        first = self.assert_child(self.launch())
        self.assertEqual(first['fixture_version'], '1.0')
        self.assertTrue((self.venv / 'pyvenv.cfg').is_file())
        self.assertTrue(self.install_calls())
        self.reset_observations()

        self.assert_child(self.launch())

        self.assertEqual(self.install_calls(), [])
        dry_runs = [call for call in self.pip_calls() if '--dry-run' in call]
        self.assertTrue(dry_runs)
        self.assertTrue(all('--no-index' in call for call in dry_runs))
        self.assertTrue(any('check' in call for call in self.pip_calls()))
        self.assertFalse(any('--upgrade' in call for call in self.pip_calls()))

    def test_uninstalled_dependency_is_restored_from_local_wheel(self):
        self.assert_child(self.launch())
        removed = subprocess.run(
            [str(self.venv / 'bin/python'), '-m', 'pip', 'uninstall', '-y', FIXTURE_NAME],
            env=self.env, text=True, capture_output=True, timeout=30,
        )
        self.assertEqual(removed.returncode, 0, removed.stderr)
        self.reset_observations()

        self.assert_child(self.launch())

        self.assertTrue(self.install_calls())

    def test_changed_requirement_installs_the_required_version(self):
        self.assert_child(self.launch())
        self.requirements.write_text(f'{FIXTURE_NAME}==2.0\n', encoding='utf-8')
        self.reset_observations()

        output = self.assert_child(self.launch())

        self.assertEqual(output['fixture_version'], '2.0')
        self.assertTrue(self.install_calls())

    def test_corrupt_import_fails_before_child_and_preserves_environment(self):
        self.assert_child(self.launch())
        package_paths = list(self.venv.glob('lib/python*/site-packages/aiohttp/__init__.py'))
        self.assertEqual(len(package_paths), 1)
        package_paths[0].write_text('raise ImportError("broken offline fixture")\n', encoding='utf-8')
        self.reset_observations()

        result = self.launch()

        self.assert_setup_failed(result)
        self.assertTrue(package_paths[0].exists())
        self.assertTrue((self.venv / 'pyvenv.cfg').exists())
        self.assertIn('KEYCHECKER_VENV', result.stderr)

    def test_dependency_install_failure_never_starts_child(self):
        self.requirements.write_text('keychecker-missing-offline-fixture==999.0\n', encoding='utf-8')

        result = self.launch()

        self.assert_setup_failed(result)
        self.assertTrue(self.install_calls())
        self.assertTrue((self.venv / 'pyvenv.cfg').exists())

    def test_custom_environment_preserves_cwd_arguments_stdin_and_exit_code(self):
        custom_venv = self.root / 'custom environment with spaces'
        relative_file = 'input fixture with spaces.txt'
        (self.caller / relative_file).write_text('local fixture content', encoding='utf-8')
        arguments = ['-file', relative_file, 'argument with spaces', 'literal;$HOME', '']
        input_text = 'first fixture line\nsecond fixture line\n'

        result = self.launch(
            *arguments, stdin=input_text,
            overrides={'KEYCHECKER_VENV': str(custom_venv), 'TEST_CHILD_EXIT': '17'},
        )

        output = self.assert_child(result, status=17, venv=custom_venv)
        self.assertEqual(output['argv'], arguments)
        self.assertEqual(Path(output['cwd']), self.caller)
        self.assertEqual(output['stdin'], input_text)
        self.assertEqual(output['file_content'], 'local fixture content')
        self.assertFalse(self.venv.exists())

    def test_invalid_custom_python_fails_without_creating_environment(self):
        result = self.launch(overrides={'KEYCHECKER_PYTHON': str(self.root / 'missing python')})

        self.assert_setup_failed(result)
        self.assertFalse(self.venv.exists())
        self.assertIn('KEYCHECKER_PYTHON', result.stderr)
        self.assertEqual(self.pip_calls(), [])

    def test_existing_nonvenv_and_broken_venv_are_preserved(self):
        for kind in ('ordinary directory', 'broken virtual environment'):
            with self.subTest(kind=kind):
                existing = self.root / kind
                existing.mkdir()
                sentinel = existing / 'preserve-me.txt'
                sentinel.write_text('preserve this existing directory', encoding='utf-8')
                if kind == 'broken virtual environment':
                    (existing / 'pyvenv.cfg').write_text('home = /missing/python\n', encoding='utf-8')
                    (existing / 'bin').mkdir()
                    interpreter = existing / 'bin/python'
                    interpreter.write_text('#!/bin/sh\nexit 23\n', encoding='utf-8')
                    interpreter.chmod(0o755)
                before = {str(path.relative_to(existing)): path.read_bytes()
                          for path in existing.rglob('*') if path.is_file()}

                result = self.launch(overrides={'KEYCHECKER_VENV': str(existing)})

                self.assert_setup_failed(result)
                after = {str(path.relative_to(existing)): path.read_bytes()
                         for path in existing.rglob('*') if path.is_file()}
                self.assertEqual(after, before)
                self.assertEqual(self.pip_calls(), [])

    def test_missing_python_bootstraps_with_mock_apt_once(self):
        commands = self.isolated_commands()
        apt_log = self.root / 'mock apt calls'
        installed = self.root / 'mock python installed'
        self.env.update({'TEST_APT_LOG': str(apt_log), 'TEST_PYTHON_INSTALLED': str(installed)})
        self.fake_command(commands, 'sudo', 'exec "$@"\n')
        self.fake_command(commands, 'apt-get', '''\
            printf '%s\\n' "$*" >> "$TEST_APT_LOG"
            if [[ "$1" == install ]]; then
                printf installed > "$TEST_PYTHON_INSTALLED"
            fi
            ''')
        self.fake_command(commands, 'python3', '''\
            [[ -f "$TEST_PYTHON_INSTALLED" ]] || exit 127
            exec "$TEST_BASE_PYTHON" "$@"
            ''')

        self.assert_child(self.launch())

        self.assertTrue(installed.exists())
        expected = ['update', 'install -y python3 python3-venv']
        self.assertEqual(apt_log.read_text().splitlines(), expected)
        self.reset_observations()
        self.assert_child(self.launch())
        self.assertEqual(apt_log.read_text().splitlines(), expected)
        self.assertEqual(self.install_calls(), [])

    def test_ready_python311_is_preferred_to_python312_missing_ensurepip(self):
        commands = self.isolated_commands()
        selected = self.root / 'selected python calls'
        apt_log = self.root / 'unexpected mock apt calls'
        self.env.update({'TEST_PYTHON_SELECTION': str(selected), 'TEST_APT_LOG': str(apt_log)})
        self.fake_command(commands, 'sudo', 'exec "$@"\n')
        self.fake_command(commands, 'apt-get', '''\
            printf '%s\\n' "$*" >> "$TEST_APT_LOG"
            exit 88
            ''')
        self.fake_command(commands, 'python3.12', '''\
            printf '3.12: %s\\n' "$*" >> "$TEST_PYTHON_SELECTION"
            if [[ "$1" == -c && "$2" == 'import venv, ensurepip' ]]; then
                exit 1
            fi
            exec "$TEST_BASE_PYTHON" "$@"
            ''')
        self.fake_command(commands, 'python3.11', '''\
            printf '3.11: %s\\n' "$*" >> "$TEST_PYTHON_SELECTION"
            exec "$TEST_BASE_PYTHON" "$@"
            ''')

        self.assert_child(self.launch())

        calls = selected.read_text().splitlines()
        self.assertTrue(any(line.startswith('3.11: -m venv ') for line in calls))
        self.assertFalse(any(line.startswith('3.12: -m venv ') for line in calls))
        self.assertFalse(apt_log.exists())


if __name__ == '__main__':
    unittest.main()
