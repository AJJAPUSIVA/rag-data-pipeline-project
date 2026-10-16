import json
import tempfile
import unittest
from pathlib import Path
from indic_data_pipeline import normalize, process, sha256


class PipelineTests(unittest.TestCase):
    def test_unicode_normalization_and_exact_duplicate(self):
        self.assertEqual(normalize('తెలుగు  భాష\n'), 'తెలుగు భాష')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'input.jsonl'
            items = [
                {'text': 'తెలుగు  భాష', 'language': 'te'},
                {'text': 'తెలుగు భాష', 'language': 'te'},
                {'text': 'हिंदी भाषा', 'language': 'hi'},
                {'text': '  ', 'language': 'te'},
            ]
            source.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in items), encoding='utf-8')
            benchmark = root / 'exclude.sha256'
            benchmark.write_text(sha256(normalize('हिंदी भाषा')) + '\n', encoding='utf-8')
            result = process(source, root / 'out.jsonl', root / 'manifest.json', root / 'state.sqlite',
                             'sample', 'CC0-1.0', benchmark)
            self.assertEqual(result['counts'], {'read': 4, 'accepted': 1, 'empty': 1, 'duplicates': 1, 'benchmark_excluded': 1})
            self.assertEqual(len((root / 'out.jsonl').read_text(encoding='utf-8').splitlines()), 1)
            self.assertEqual(result['language_counts'], {'te': 1})

    def test_unknown_licence_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(ValueError):
                process(root / 'input.jsonl', root / 'out.jsonl', root / 'manifest.json', root / 'state.sqlite', 'sample', 'unknown')


if __name__ == '__main__':
    unittest.main()
