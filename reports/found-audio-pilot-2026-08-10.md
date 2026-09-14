# Found-audio acquisition pilot — Turkish single-speaker sources

Status: completed 2026-08-10. Verdict: **no-go for now**; the Voice B campaign is the data path.

## What was tested

Bounded metadata-and-sample pilot of the two commercially-clean source families proposed in the
campaign brief discussion: LibriVox Türkçe (public domain) and VOA Türkçe (US-government public
domain). No bulk downloading; one sample probe.

## Findings

### LibriVox Türkçe: zero usable inventory

- The LibriVox API's `language=Turkish` filter silently returns the general (English-dominated)
  catalog — the initial "~hundreds of hours" impression was an API artifact, not real inventory.
- Authoritative check against `archive.org` (`collection:librivoxaudio AND language:(Turkish)`)
  returns **0 items**. LibriVox has effectively no Turkish holdings. This corrects the earlier
  estimate in the campaign discussion, which was wrong.

### VOA Türkçe: small, multi-speaker, below quality bar

- Discoverable audio RSS inventory: one zone ("VOA Türkçe Deprem Bölgesinde"), 10 episodes,
  **1.63 h listed**. The main podcast feed is a video product and did not serve parseable RSS.
- Sample probe (7.9 MB episode): 44.1 kHz mono, clipping 0.0, **estimated SNR 32.6 dB** —
  respectable broadcast audio but well below the ≥45 dB the accepted training corpus averages,
  and the content is field reporting: multi-speaker, ambient noise, interview turns.
- Larger VOA archives exist on-site outside RSS, but the accessible-by-pipeline inventory measured
  here is under two hours of the wrong style.

## Verdict

- Public-domain Turkish single-speaker audio at scale does not exist in the two clean sources.
- The remaining lane — YouTube CC-BY filtered channels — requires per-channel human curation and
  attribution tracking; it cannot be piloted mechanically and should only be revisited if the
  foundation (currently 47 h and not the bottleneck) ever becomes the limiting factor.
- Effort is better spent on the consented Voice B campaign
  (`configs/collection/candidate-b-scripted-v2.json`), where every hour is on-style, single-speaker,
  studio-quality, and rights-clean.

## Evidence trail

- LibriVox API calls (query + path-style + strict field filter) all returned non-Turkish catalogs;
  archive.org advanced search returned `numFound: 0`.
- VOA feeds probed via iTunes podcast directory + direct RSS; sample audio analyzed with the
  project's `analyze_signal` (same signal gate used for corpus acceptance).
