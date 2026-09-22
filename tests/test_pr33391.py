"""Keep PR #33391 logic/tests unchanged; the only behavioral addition wires native IPC.

Ported from the eval suite:test_sglang_pr33391.py; the upstream fixture lives in
tests/fixtures/. The recipe-order assertion now reads profile.yaml instead of
the serving profile.
"""
import ast
from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / 'patches'
FIXTURES = Path(__file__).resolve().parent / 'fixtures'
PROFILE = yaml.safe_load((ROOT / 'profile.yaml').read_text())


def additions(text, target):
    active = False
    result = []
    for line in text.splitlines():
        if line.startswith('+++ b/'):
            active = line[6:] == target
        elif line.startswith('--- ') or line.startswith('diff --git '):
            active = False
        elif active and line.startswith('+'):
            result.append(line[1:])
    return '\n'.join(result) + '\n'


def functions_only(source):
    start = source.index('    def materialize_cpu_views_for_pickle(')
    return ast.dump(ast.parse(source[:start] + 'class Holder:\n' + source[start:]), include_attributes=False)


class FaithfulPRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = (FIXTURES / 'sglang-pr33391-upstream.patch').read_text()
        cls.adapted = (PATCHES / 'sglang-33391-mm-pickle-storage.patch').read_text()

    def test_compaction_helper_and_all_methods_match_upstream(self):
        path = 'python/sglang/srt/managers/schedule_batch.py'
        self.assertEqual(functions_only(additions(self.upstream, path)),
                         functions_only(additions(self.adapted, path)))

    def test_runtime_rebase_does_not_delete_existing_code(self):
        allowed = {
            'python/sglang/srt/managers/io_struct.py',
            'python/sglang/srt/managers/schedule_batch.py',
            'test/registered/unit/managers/test_mm_utils_split.py',
        }
        changed = set()
        current = None
        for line in self.adapted.splitlines():
            if line.startswith('--- a/'):
                current = None
            elif line.startswith('+++ b/'):
                current = line[6:]
                changed.add(current)
            elif current and current.startswith('python/'):
                self.assertFalse(line.startswith('-'),
                                 f'Unexpected runtime deletion in {current}')
        self.assertEqual(changed, allowed)

    def test_added_source_has_no_trailing_whitespace(self):
        # Patch context lines can consist of one space; added source cannot.
        for patch in (self.upstream, self.adapted):
            for line in patch.splitlines():
                if line.startswith('+') and not line.startswith('+++'):
                    self.assertEqual(line, line.rstrip())

    def test_upstream_added_cpu_tests_are_unchanged(self):
        path = 'test/registered/unit/managers/test_mm_utils_split.py'
        self.assertEqual(additions(self.upstream, path), additions(self.adapted, path))

    def test_profile_declares_full_series_and_baked_gate(self):
        self.assertEqual([p['file'] for p in PROFILE['patches']], [
            'sglang-glm53-integration.patch', 'sglang-glm53-spark-mxfp8.patch',
            'sglang-35609-batch-is-full.patch', 'sglang-33391-mm-pickle-storage.patch',
            'sglang-glm53-ssd-hicache.patch'])
        # The gate ships inside the image; no mount, no external script.
        dockerfile = (ROOT / 'docker' / 'Dockerfile').read_text()
        self.assertIn('COPY gates/ ${GLM_GATES}/', dockerfile)

    def test_new_gate_runs_upstream_file_and_both_send_paths(self):
        tree = ast.parse((ROOT / 'gates' / 'gate-pr33391.py').read_text())
        source = ast.unparse(tree)
        for text in ('test/registered/unit/managers/test_mm_utils_split.py',
                     'io.sock_send', 'io.async_sock_send', 'io.BatchTokenizedGenerateReqInput'):
            self.assertIn(text, source)
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        self.assertIn('pr33391_main()', ast.unparse(main))


if __name__ == '__main__':
    unittest.main()
