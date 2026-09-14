# Antalia 1 technical report

LaTeX source for the release report. Primary arXiv category: `eess.AS`.

- `main.tex` — the report (article class, two-column, natbib/plainnat)
- `references.bib` — bibliography
- `blog.md` — companion Hugging Face blog post

## Build

With a TeX Live installation:

```sh
cd paper
latexmk -pdf main.tex
```

`latexmk` runs pdflatex and bibtex as many times as needed and writes `main.pdf`.
Clean up with `latexmk -c`.

Without TeX Live, [Tectonic](https://tectonic-typesetting.github.io/) builds it in one command
and downloads the required packages on first run:

```sh
cd paper
tectonic main.tex
```

## Before submitting to arXiv

The report is complete and compiles; it has not been submitted. Primary category: `eess.AS`.

Once an arXiv identifier exists, replace the interim paper link
(`https://github.com/0daycloud/antalia/blob/main/paper/main.pdf`) with the arXiv URL in:

- this repository's `README.md` and `pyproject.toml`
- `site/index.html` and `site/tr/index.html`
- the Hugging Face model cards for `cloud0day3/antalia-1` and `cloud0day3/antalia-1-foundation`
  (English and Turkish) and the `cloud0day3/antalia-eval` dataset card
- the `note` field of the BibTeX entry in each of the above
