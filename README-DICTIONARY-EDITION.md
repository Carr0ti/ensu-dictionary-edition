# Ensu Dictionary Edition — first database build

This starter adds a **lossless first-stage importer**. It does not yet modify Ensu's Android UI or semantic-pack catalog.

## What it does

- Streams the Kaikki/Wiktextract JSONL gzip source.
- Selects records identified as English (`lang_code=en` or `lang=English`).
- Stores the complete JSON object in `entries.raw_json`.
- Adds indexed columns for headword, normalized headword, language and part of speech.
- Builds a full-text index as a convenience; the full JSON remains the authoritative record.
- Includes a test proving that multiple glosses, examples, and extra fields survive.

The source is Wiktionary-derived data, not Oxford, Cambridge, or Merriam-Webster proprietary dictionary text. Before publishing a database or pack, check and include the current source license and attribution requirements.

## Upload to GitHub on Android

1. Open your repository and switch to branch `dictionary-edition-dev`.
2. Open **Add file → Upload files**.
3. Upload the files and folders from this ZIP while preserving their paths:
   - `.github/workflows/build-dictionary-database.yml`
   - `scripts/ingest_wiktextract.py`
   - `scripts/__init__.py`
   - `tests/test_ingest.py`
   - `README-DICTIONARY-EDITION.md`
4. Commit directly to `dictionary-edition-dev`.

If the GitHub mobile page does not let you upload a folder, create the folders first and upload each file into its matching folder. Keep the leading dot in `.github`.

## First safe test

1. Open the **Actions** tab.
2. Select **Build complete dictionary database**.
3. Tap **Run workflow**.
4. Keep `smoke_test_limit` as `1000` for the first test.
5. Run it and wait for the green check.
6. Open the completed run and download the artifact named `ensu-dictionary-sqlite`.

The first test downloads the source stream but only stores 1,000 English records. This confirms the importer and artifact pipeline; it is not the full dictionary.

## Full database build

After the 1,000-record test passes, run the workflow again and clear the `smoke_test_limit` input completely. An empty value means full import. The workflow has a six-hour job timeout. If the full source download/import exceeds that limit, we'll add resumable source caching and chunked database building rather than reducing dictionary content.

## Important limitation

This first milestone creates the complete source-preserving SQLite database. It does not yet create the Ensu semantic retrieval pack, add an exact-lookup screen, or build an APK. Those are later milestones after validating the database.
