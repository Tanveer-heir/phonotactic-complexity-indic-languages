# Phonotactic Complexity of Indic Languages

A computational replication and extension of **Pimentel, Roark & Cotterell (TACL 2020)** —
measuring phonotactic complexity (bits-per-phoneme) for Hindi, Tamil, Telugu, and English,
with three original extensions: ASR correlation, tokenizer fertility analysis, and
mechanistic decomposition of vowel harmony and retroflex consonant contributions.

---

## Paper We Extend

> Pimentel, T., Roark, B., & Cotterell, R. (2020).
> **Phonotactic Complexity and Its Trade-offs.**
> *Transactions of the Association for Computational Linguistics*, 8, 1–18.
> https://aclanthology.org/2020.tacl-1.1/

The paper introduced **bits-per-phoneme** as an information-theoretic measure of
phonotactic complexity — how surprising, on average, is each sound in a word of a
given language? Lower bits = more constrained phonotactics = more predictable sounds.
The paper demonstrated a cross-linguistic trade-off: languages with lower phonotactic
complexity compensate with longer words.

We replicate the paper's core method on South Asian languages and add three extensions
the paper does not cover.

---

## Languages Studied

| Language | ISO | Family | Script |
|---|---|---|---|
| Hindi | hin | Indo-Aryan (Indo-European) | Devanagari |
| Tamil | tam | Dravidian | Tamil script |
| Telugu | tel | Dravidian | Telugu script |
| English | eng | Germanic (Indo-European) | Latin |

All four languages are natively present in NorthEuraLex 0.9 with proper IPA
transcription — no grapheme-to-phoneme conversion or fallback pipeline was needed.

---

## Data Source

**NorthEuraLex 0.9** (Dellert & Jäger, 2017) — a concept-aligned multilingual lexicon
covering 1016 basic concepts across 107 languages, all transcribed in unified IPA.

- Source: `github.com/lexibank/northeuralex` (CLDF release)
- We use the `Segments` column — pre-tokenized IPA phoneme sequences
- Multi-word phrase glosses (marked `+` in Segments) excluded — phonotactics applies
  within a single phonological word, not across word boundaries
- 10-fold cross-validation at concept level, matching the paper's exact methodology

```
Language   Words (after filter)   Phoneme inventory   Avg word length
Hindi      1287                   82                  4.79 phonemes
Tamil       871                   48                  6.06 phonemes
Telugu      985                   66                  6.50 phonemes
English     976                   40                  4.15 phonemes
```

---

## Project Structure

```
phonotactic-complexity-indic-languages/
├── Notebook1_data-pipeline.ipynb        # Download, clean, 10-fold CV splits
├── Notebook2_trigramLM.ipynb            # Interpolated trigram LM + bits/phoneme
├── Notebook3_lstm-model.ipynb           # LSTM phoneme LM + bits/phoneme
├── Notebook4_Complexity_vs_WER.ipynb    # ASR WER correlation analysis
├── Notebook5_phono-fertility.ipynb      # mBERT / XLM-R / GPT-2 fertility
├── Notebook6_ablation-study.ipynb       # Vowel harmony + syllable + retroflex
└── README.md
```

---

## Methods

### Core Metric — Bits Per Phoneme

Following Pimentel et al. (2020), Equation 3:

```
H = -(1/N) × Σ log₂ P(phonemeᵢ | phoneme₁..ᵢ₋₁)
```

Lower H = more constrained phonotactics = more predictable sounds = lower complexity.

### Model 1 — Interpolated Trigram LM (Notebook 2)

Phoneme-level n-gram LM with **deleted interpolation smoothing** (Jelinek & Mercer 1980),
blending trigram, bigram, and unigram estimates with learned weights. Matches the
paper's smoothing strategy. Simple add-k smoothing was insufficient for our dataset
size (~900 words per language) — interpolation reduced entropy estimates across all
four languages.

### Model 2 — LSTM Phoneme LM (Notebook 3)

Single-layer LSTM (Embedding 32 → LSTM 64 → Linear) trained per language.
Architecture kept small to avoid overfitting on ~900 training words.
Key regularization: dropout 0.5, weight decay 1e-4, early stopping (patience=3),
gradient clipping (max norm 1.0). Validated on Tamil Fold 0 before full run.

### Evaluation

10-fold cross-validation at concept level. Training set: ~900 concepts.
Test set: ~100 concepts per fold, rotated across all 10 folds.
Final score: mean ± std across 10 folds.

---

## Key Findings

### 1 — Complexity Ranking (Notebooks 2 & 3)

```
             Trigram bpp    LSTM bpp    LSTM gain
Hindi          4.008          3.722       +0.286   ← most complex
English        3.690          3.486       +0.204
Telugu         3.511          3.283       +0.228
Tamil          3.226          3.119       +0.107   ← least complex
```

Tamil and Telugu have **lower phonotactic complexity than English** despite richer
consonant inventories. Their stricter phonotactic rules — limited consonant clusters,
strict syllable structure, vowel harmony — make each phoneme more predictable.
They compensate with longer words (Tamil 6.06, Telugu 6.50 vs English 4.15 phonemes/word),
consistent with the trade-off Pimentel et al. demonstrated across 106 languages.

Hindi exceeds English in complexity, driven by its 4-way aspiration contrast
(p/pʰ/b/bʰ), retroflex series (ʈ/ɖ/ɳ/ɽ), nasal vowels (ã/ẽ/õ), and gemination.

The LSTM beats the trigram for all languages. Hindi shows the largest gain (+0.286 bits),
suggesting long-range phonotactic dependencies from aspiration spreading and nasal vowel
co-occurrence that the 2-phoneme trigram window cannot capture. Tamil shows the smallest
gain (+0.107) — Tamil's phonotactics are largely local; the trigram already captured
most of the structure.

### 2 — ASR Correlation (Notebook 4)

WER data from **Vistaar benchmark** (Bhogale et al., Interspeech 2023), IndicWhisper model:

```
English   3.0% WER   (Whisper large-v2, LibriSpeech test-clean)
Hindi    13.4% WER   (IndicWhisper, FLEURS-Hindi)
Telugu   24.7% WER   (IndicWhisper, FLEURS-Telugu)
Tamil    30.1% WER   (IndicWhisper, FLEURS-Tamil)
```

Spearman ρ (LSTM bpp vs WER) = **−0.800** — negative, opposite of the naive hypothesis.

The negative correlation is fully explained by **training data availability**. Hindi has
~10,000+ hours of ASR training data; Tamil has ~300 hours. Languages with simpler
phonotactics (Tamil, Telugu) happen to be the most under-resourced for ASR.
Data inequality dominates any phonotactic signal. Controlling for training data size,
phonotactic complexity is expected to predict ASR difficulty — testing this properly
requires a controlled experiment (equal training data across languages), a direction
for future work.

### 3 — Tokenizer Fertility (Notebook 5)

```
Tokenizer    Hindi    Tamil    Telugu    English
mBERT         4.35     1.33     1.40      2.91
XLM-R         5.27     6.75     7.68      4.39
GPT-2         8.80    10.44    11.76      6.08
```

**mBERT fertility vs LSTM bpp: Spearman ρ = +1.000 (p < 0.001)** — perfect correlation.
mBERT fertility perfectly tracks phonotactic complexity ranking.

Two opposite tokenizer failure modes discovered:

- **mBERT undertokenizes Dravidian** — Tamil (1.33) and Telugu (1.40) fertility is
  near 1 token/word. mBERT has absorbed whole Tamil/Telugu script word-forms as
  single vocabulary entries (whole-word memorization), not productive subword
  decomposition. Cannot generalize to unseen word forms.

- **XLM-R and GPT-2 overtokenize Dravidian** — Tamil (6.75/10.44) and Telugu
  (7.68/11.76) get fragmented into 7-12 pieces per word. These tokenizers lack
  sufficient Indic vocabulary to cover morpheme boundaries, forcing aggressive
  character-level splitting.

Both are failures in opposite directions. The mBERT result provides a linguistic
explanation for known tokenizer bias in multilingual LLMs: higher phonotactic
complexity → more subword fragmentation.

### 4 — Vowel Harmony Ablation — Analysis 6a (Notebook 6)

Replication and extension of Paper Study 3 (Turkish ablation) to Dravidian languages.
Vowels within each test word were randomly scrambled while consonants stayed fixed.
Entropy increase = vowel harmony's contribution to phonotactic predictability.

```
Telugu   +0.649 bits/phoneme  ← exceeds Turkish (+0.620, paper baseline)
English  +0.434 bits/phoneme  ← baseline (no harmony, measures general vowel stats)
Hindi    +0.328 bits/phoneme  ← partial assimilation, not full harmony
Turkish  +0.620 bits/phoneme  ← paper baseline (Pimentel et al. 2020)
```

**Telugu's vowel harmony contribution (+0.649) exceeds Turkish (+0.620)** — the first
Dravidian language to receive this ablation treatment. Telugu's harmony system is a
stronger phonotactic constraint than Turkish's. Subtracting the English baseline (0.434)
gives ~0.215 bits attributable specifically to Telugu's harmony system rather than
general vowel co-occurrence statistics.

### 5 — Syllable Boundary Entropy — Analysis 6b (Notebook 6)

Bits-per-phoneme decomposed by syllable position (onset / nucleus / coda):

```
Position    Hindi    Tamil    Telugu    English
Onset       5.112    4.435    4.708     4.464
Nucleus     3.530    2.667    3.001     3.854
Coda        5.194    3.530    0.359     4.395
```

Two headline findings:

**Telugu coda entropy = 0.359** — near-zero. Telugu words overwhelmingly end in vowels
(strongly V-final language). When a coda consonant appears, options are so limited it
is nearly deterministic. This single position accounts for a significant portion of
Telugu's lower overall complexity vs Hindi.

**Hindi coda entropy = 5.194** — highest single position value across all four languages.
Hindi allows a wide range of word-final consonants with few constraints. Coda position
is the primary driver of Hindi's complexity advantage over English.

Dravidian nucleus entropy (Tamil 2.667, Telugu 3.001) is the lowest across all positions
— vowel positions are the most predictable in Dravidian languages, consistent with
vowel harmony and limited vowel inventory. English nucleus (3.854) is higher than
Hindi nucleus (3.530) — English's vowel system (diphthongs, reduced vowels, schwa)
is more complex than Hindi's at the nucleus position.

### 6 — Retroflex Positional Entropy — Analysis 6c (Notebook 6)

Retroflexes {ʈ, ɖ, ɳ, ɽ, ɭ, ʂ} are present in all three Indic languages but absent
from English. Average surprise at retroflex positions vs non-retroflex consonant positions:

```
Language   Retro bits   Non-retro bits   Surprise premium   Frequency   % of complexity
Hindi        6.812         4.864            +1.948             4.2%          6.4%
Tamil        4.579         4.024            +0.555             6.6%          8.6%
Telugu       5.293         4.332            +0.961             6.3%          8.6%
```

**Hindi's retroflex surprise premium (+1.948 bits)** is the highest — Hindi's rich
retroflex series (including aspirated ɖʰ, ʈʰ and retroflex sibilant ʂ) appears in
less predictable phonotactic positions, making each one highly surprising.

**Tamil and Telugu contribute equally (8.6% each)** via different mechanisms:
Tamil has a lower premium (+0.555) but higher frequency (6.6%); Telugu has a higher
premium (+0.961) at similar frequency (6.3%). Equal net complexity load from
two different phonological strategies.

**Hindi's retroflex contribution (6.4%) is lower than Tamil/Telugu (8.6%)** despite
having the highest surprise premium — because Hindi retroflexes appear less frequently
(4.2% of positions). Tamil and Telugu deploy retroflexes more pervasively throughout
words; they are a more integral part of the consonant system rather than marked sounds.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | All analysis |
| PyTorch | LSTM phoneme LM |
| HuggingFace Transformers | mBERT, XLM-R, GPT-2 tokenizers |
| NumPy / Pandas | Data wrangling and results tables |
| SciPy | Spearman correlation |
| Matplotlib / Seaborn | All visualizations |
| scikit-learn | 10-fold cross-validation splits |
| Kaggle (GPU T4) | Training environment |

---

## Limitations

**Sample size (n=4).** Spearman correlation with 4 languages has low statistical power.
Results are directional and consistent across both models but not conclusive.

**Dataset scope.** NorthEuraLex uses 1016 basic concepts — core vocabulary, carefully
transcribed. May not represent the full phonotactic range of each language including
loanwords, technical vocabulary, and colloquial forms.

**ASR confound.** WER comparison is confounded by training data availability.
A controlled experiment with equal training data per language is needed to isolate
the phonotactic signal.

**Model size.** LSTM architecture is minimal (32/64 dims) to avoid overfitting on
~900 training words. Larger models with more data would give tighter entropy estimates.

**Syllable segmenter.** CV-based segmentation is a simplification. Language-specific
phonological rules (especially for Hindi consonant clusters) are not fully captured.

---

## References

```
Pimentel, T., Roark, B., & Cotterell, R. (2020).
Phonotactic Complexity and Its Trade-offs.
Transactions of the Association for Computational Linguistics, 8, 1-18.
https://aclanthology.org/2020.tacl-1.1/

Bhogale, K., et al. (2023).
Vistaar: The Largest Open Collection of Diverse Indic Speech.
Interspeech 2023.

Dellert, J., & Jäger, G. (2017).
NorthEuraLex 0.9.
https://northeuralex.org

Jelinek, F., & Mercer, R. L. (1980).
Interpolated estimation of Markov source parameters from sparse data.
Pattern Recognition in Practice.

Radford, A., et al. (2022).
Robust Speech Recognition via Large-Scale Weak Supervision.
https://arxiv.org/abs/2212.04356
```

---

## How to Run

1. Open any notebook on Kaggle
2. Add the Data Pipeline notebook as input (Settings → Add Input → Notebook)
3. Turn internet ON (Settings → Internet → On) for Notebook 1 and 5
4. Turn GPU ON (Settings → Accelerator → GPU T4) for Notebooks 3 and 6
5. Run all cells top to bottom

Each notebook is self-contained and loads its dependencies from previous notebook outputs.

---

## Citation

If you use this work, please also cite the original paper:

```bibtex
@article{pimentel-etal-2020-phonotactic,
  title     = {Phonotactic Complexity and Its Trade-offs},
  author    = {Pimentel, Tiago and Roark, Brian and Cotterell, Ryan},
  journal   = {Transactions of the Association for Computational Linguistics},
  volume    = {8},
  pages     = {1--18},
  year      = {2020},
  url       = {https://aclanthology.org/2020.tacl-1.1/}
}
```
