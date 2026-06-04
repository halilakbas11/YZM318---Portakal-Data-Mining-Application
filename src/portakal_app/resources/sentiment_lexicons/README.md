# Sentiment lexicon resources (SentiART / LiLaH)

The Sentiment Analysis widget's **SentiART** and **LiLaH** methods read Orange-compatible
`.pickle` lexicon files from this folder. This is the **highest-priority** location the
widget searches (see `candidate_sentiment_resource_dirs` in
`src/portakal_app/ui/screens/sentiment_analysis_screen.py`), so files placed here are
used offline and take precedence over the auto-download cache in
`~/.portakal/sentiment_lexicons/`.

## Expected files

| Method   | Language            | Filename             |
|----------|---------------------|----------------------|
| SentiART | English             | `SentiArt_EN.pickle` |
| SentiART | German              | `SentiArt_DE.pickle` |
| LiLaH    | Croatian            | `LiLaH-HR.pickle`    |
| LiLaH    | Dutch               | `LiLaH-NL.pickle`    |
| LiLaH    | Slovenian           | `LiLaH-SL.pickle`    |

These are **not committed** to the repo (large binaries). If they are missing the widget
shows zeros (0.000) for SentiART/LiLaH columns and a status warning.

## How to populate (one-time, needs internet)

From the repository root, run:

```bash
python tools/fetch_sentiment_lexicons.py
```

This downloads the official files from Orange/Biolab
(`https://file.biolab.si/files/sentiart/` and `https://file.biolab.si/files/sentiment-lilah/`)
straight into this folder and verifies each one loads as a word→scores mapping.
After that, SentiART/LiLaH work fully offline.

## Important usage note

LiLaH lexicons only cover **Croatian / Dutch / Slovenian**, so running LiLaH on English
text yields ~0 scores even when the pickle is present (almost no tokens match). For
English documents use **SentiART (English)** or the VADER / Liu & Hu methods instead.

## Security note

`.pickle` files are executed by Python on load. Only use the official Orange/Biolab files
(this mirrors what Orange Text Mining itself downloads).
