import gzip
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.ingest_wiktextract import import_data

class ImportTests(unittest.TestCase):
    def test_preserves_full_record_and_filters_language(self):
        records = [
            {
                "word": "bright", "lang": "English", "lang_code": "en", "pos": "adj",
                "senses": [
                    {"glosses": ["giving out or reflecting much light", "intelligent"],
                     "examples": [{"text": "A bright room."}, {"text": "A bright child."}],
                     "tags": ["positive"]},
                    {"glosses": ["vivid in colour"], "raw_glosses": ["very bright"]}
                ],
                "sounds": [{"ipa": "/braɪt/"}],
                "etymology_text": "From Middle English.",
                "synonyms": [{"word": "shining"}],
                "custom_future_field": {"keep": True}
            },
            {"word": "lumière", "lang": "French", "lang_code": "fr", "pos": "noun",
             "senses": [{"glosses": ["light"]}]}
        ]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "sample.jsonl.gz"
            output = Path(tmp) / "dictionary.sqlite3"
            with gzip.open(source, "wt", encoding="utf-8") as f:
                for item in records:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            stats = import_data(str(source), output, None, 10)
            self.assertEqual(stats["english_records_stored"], 1)
            db = sqlite3.connect(output)
            raw = db.execute("SELECT raw_json FROM entries WHERE word='bright'").fetchone()[0]
            saved = json.loads(raw)
            self.assertEqual(len(saved["senses"]), 2)
            self.assertEqual(len(saved["senses"][0]["glosses"]), 2)
            self.assertEqual(len(saved["senses"][0]["examples"]), 2)
            self.assertEqual(saved["custom_future_field"], {"keep": True})
            self.assertEqual(db.execute("SELECT count(*) FROM entries").fetchone()[0], 1)
            db.close()

if __name__ == "__main__":
    unittest.main()
