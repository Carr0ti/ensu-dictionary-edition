import gzip
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.ingest_wiktextract import import_data

class ImportTests(unittest.TestCase):
    def make_source(self, folder, records):
        source = Path(folder) / "sample.jsonl.gz"
        with gzip.open(source, "wt", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return source

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
            source = self.make_source(tmp, records)
            output = Path(tmp) / "dictionary.sqlite3"
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

    def test_resume_continues_without_duplicate_entries(self):
        records = [
            {"word": "alpha", "lang": "English", "lang_code": "en", "pos": "noun"},
            {"word": "beta", "lang": "English", "lang_code": "en", "pos": "noun"},
            {"word": "gamma", "lang": "English", "lang_code": "en", "pos": "noun"},
            {"word": "bonjour", "lang": "French", "lang_code": "fr", "pos": "noun"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            source = self.make_source(tmp, records)
            output = Path(tmp) / "dictionary.sqlite3"

            first = import_data(
                str(source), output, None, 1,
                checkpoint_source_records=1,
                stop_after_source_records=2,
            )
            self.assertEqual(first["english_records_stored"], 2)

            second = import_data(
                str(source), output, None, 1,
                resume=True,
                checkpoint_source_records=1,
            )
            self.assertEqual(second["english_records_stored"], 3)

            db = sqlite3.connect(output)
            words = [row[0] for row in db.execute(
                "SELECT word FROM entries ORDER BY id"
            )]
            self.assertEqual(words, ["alpha", "beta", "gamma"])
            self.assertEqual(
                db.execute("SELECT value FROM import_state WHERE key='complete'").fetchone()[0],
                "1",
            )
            db.close()

if __name__ == "__main__":
    unittest.main()
