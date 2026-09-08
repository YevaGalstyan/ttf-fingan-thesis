# Master's Thesis — Synthetic Price Path Generation for European Gas Markets

LaTeX source and implementation for the Master's thesis *Synthetic Price Path Generation for European Gas Markets Using Generative Adversarial Networks with Applications to Option Pricing*.

Yeva Galstyan · Department of Applied Computer Science, Fulda University of Applied Sciences
Supervisor: Prof. Dr. Christoph Scheich · Co-supervisor: Dr. Alexander Jungwirth

## Repository structure

```
main.tex                          Document root — includes everything below
literatur.bib                     Bibliography, auto-exported from Zotero
content/
  misc/
    preamble.tex                  Packages, formatting, thesis setup
    titlepage.tex
    abstract.tex
    appendix.tex
    ai_documentation.tex          Required documentation of AI tool usage
    declaration.tex               Declaration of authorship
  01_introduction.tex
  02_financial-fundamentals/      Chapter 2, one file per section
    index.tex                     Chapter lead-in and includes
    financial-markets.tex
    gas-markets.tex
    futures.tex
    options.tex
    implied-vol.tex
    stylized-facts.tex
  03_ml-fundamentals/             Chapter 3
    index.tex
    neural-networks.tex
    generative-adversarial-nets.tex
    related-work.tex
  04_methodology/                 Chapter 4
    index.tex
    data-pipeline.tex
    model-architecture.tex
    configuration-ablation.tex
    benchmarks.tex
    evaluation-method.tex
    pricing-setup.tex
    software-env.tex
  05_experiments/                 Chapter 5
    index.tex
    properties-ttf.tex
    baseline.tex
    network-capacity.tex
    condition-window.tex
    effect-cost.tex
    model-selection.tex
    option-prices.tex
  06_discussion.tex
  07_conclusion.tex
figures/                          Figures included in the document
assets/                           Institutional logo and template assets
code/                             Implementation — see below
```

Build artifacts (`main.aux`, `main.log`, `main.toc`, and similar) are generated on every compile and are excluded via `.gitignore`.

## Code

```
code/
  scripts/
    paths.py                        Shared input/output paths, imported by every script below
    data/
      build_front_month.py          Bloomberg parquet files to daily front-month series
      build_options_panel.py        Databento folders to contract-day master table
      build_pricing_date.py         Pricing date table for the option pricing benchmark
      diagnostics_front_month.py    Front-month series diagnostics quoted in the thesis
      filter_options_panel.py       Row/contract counts after each options panel filter
    model/
      FinGAN.py                     Fin-GAN training, adapted from the reference code
      ttf_eval.py                   Distributional evaluation of a trained generator
      run_ablation.py               Ablation grid, ten cost functions x eleven configurations
    plots/
      price_series.py               Front-month price series figure
      log_return.py                 Log return series figure
      autocorr_foregan.py           ForGAN autocorrelation-of-squared-returns figure
    results/
      ttf_statistics.py             Summary statistics of the realized return series
      summarize_runs.py             Ablation results aggregated into thesis tables
      full_grid_latex.py            Full ablation grid as LaTeX tables (content/appendix)
    exploration/
      *_audit.ipynb                 Source data audits (bbo, definitions, master, ohlc,
                                     statistics, status) underlying Sec. 4.1
  data/                           Source data — not tracked
  output/                         Generated files — not tracked
  requirements.txt
```

Every script under `scripts/` resolves its paths from `paths.py` rather than hardcoding them,
so run them from anywhere — the working directory does not matter.

The source data is not included in this repository. Place the Bloomberg parquet files and the Databento folders under `code/data/` before running anything.

```bash
cd code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` records the versions used for the reported results; the same versions are listed in the software environment section of the thesis.

## Building the document

Requires a TeX distribution with `latexmk` and `biber`. On macOS, MacTeX provides both.

```bash
latexmk -pdf main.tex
```

The bibliography uses BibLaTeX with the Biber backend, so `latexmk` runs the extra passes automatically. To clear all generated files:

```bash
latexmk -C
```

## Local setup

### TeX distribution

Install MacTeX, either from [tug.org/mactex](https://tug.org/mactex/) or via Homebrew:

```bash
brew install --cask mactex
```

The installer requires admin rights — watch for the password prompt, as the install silently does nothing if it is missed. Open a new terminal afterwards and confirm:

```bash
which latexmk biber
```

Both should resolve under `/Library/TeX/texbin/`.

### VS Code

Install the **LaTeX Workshop** extension (James Yu). Recommended additions: **Code Spell Checker** and **LTeX+** for prose checking.

Editor settings are not tracked. Create `.vscode/settings.json` locally with:

```json
{
  "latex-workshop.latex.path": "/Library/TeX/texbin",
  "latex-workshop.latex.recipe.default": "latexmk",
  "latex-workshop.latex.autoBuild.run": "onSave",
  "latex-workshop.view.pdf.viewer": "tab",
  "editor.wordWrap": "on",
  "cSpell.language": "en-US",
  "ltex.language": "en-US"
}
```

The default recipe must be `latexmk` (pdfLaTeX). The document uses Type 1 Palatino, which LuaLaTeX cannot resolve — building with the LuaLaTeX recipe fails on missing glyphs, including the euro sign.

Build with `LaTeX Workshop: Build LaTeX project` from the Command Palette; preview with `Ctrl/Cmd+Alt+V`.

### Bibliography

`literatur.bib` is auto-exported from Zotero via Better BibTeX. To reconnect on a new machine:

1. Right-click the `thesis` collection in Zotero → **Export Collection...**
2. Format: **Better BibTeX**, with **Keep updated** enabled
3. Save as `literatur.bib` in the repository root

Existing auto-exports can be managed under Zotero → Settings → Better BibTeX → Automatic export. Note that exports only run while Zotero is open.

Reference metadata is edited in Zotero, never directly in `literatur.bib` — the file is overwritten on every export.

## Conventions

- American English spelling throughout
- Citation keys follow `authorYEARkeyword`
- `\textcite` where the author is the subject of the sentence, `\parencite` otherwise
- Passive past tense for methodological decisions, present tense for architectural and definitional descriptions
- Chapters are split into per-section files under `content/`
- Skewness and excess kurtosis are reported as the biased sample estimators $g_1$ and $g_2$, applied to realized and generated returns alike

## License

The code in `code/` builds on the [Fin-GAN reference implementation](https://github.com/milenavuletic/Fin-GAN) by Vuletić et al., which is licensed under GPL-3.0, and is therefore also GPL-3.0. See `code/LICENSE`. The thesis text and figures are not covered by that
license.

## Data

The TTF futures data (Bloomberg, via BASF) and the ICE TTF options data (Databento) are licensed and are not included in this repository. Processing steps are documented in the methodology chapter and are reproducible given access to the same sources.

## Status

Draft.