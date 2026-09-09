#!/usr/bin/env python3.12
"""
Thai Word Segmentation — Full Pipeline
Generates ws_submission.csv from ws_test.txt using:
  1. PyThaiNLP newmm + longest tokenizer ensemble (chunk-based, alignment-safe)
  2. LightGBM character feature classifier trained on pseudo-labels
  3. BIO sequence post-processing
"""

import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path

from pythainlp.tokenize import word_tokenize
from pythainlp.util import is_thai_char

import lightgbm as lgb
from sklearn.metrics import f1_score, classification_report

import warnings
warnings.filterwarnings('ignore')

BASE_DIR    = Path('.')
TEST_FILE   = BASE_DIR / 'ws_test.txt'
SAMPLE_SUB  = BASE_DIR / 'ws_sample_submission.csv'
OUTPUT_FILE = BASE_DIR / 'ws_submission.csv'

# =============================================================================
# 1. LOAD TEST DATA
# =============================================================================
print("\n" + "="*60)
print("1. LOADING TEST DATA")
print("="*60)

with open(TEST_FILE, 'r', encoding='utf-8') as f:
    test_text = f.read()

print(f'Total chars (incl. whitespace): {len(test_text):,}')

non_ws_ids   = []
non_ws_chars = []
for i, c in enumerate(test_text):
    if c != ' ':
        non_ws_ids.append(i + 1)
        non_ws_chars.append(c)

print(f'Non-whitespace chars: {len(non_ws_chars):,}')

sample_df = pd.read_csv(SAMPLE_SUB)
assert len(non_ws_ids) == len(sample_df), f'Row count mismatch!'
assert list(non_ws_ids) == list(sample_df['Id']), 'ID mismatch!'
print('✅ IDs verified')

# =============================================================================
# 2. BIO HELPERS — chunk-based, handles tokenizer char-count differences
# =============================================================================

def words_to_bio(word):
    """Convert one word string to B/I/E label list."""
    n = len(word)
    if n == 1:   return ['B_WORD']
    if n == 2:   return ['B_WORD', 'E_WORD']
    return ['B_WORD'] + ['I_WORD'] * (n - 2) + ['E_WORD']


def tokenize_chunk_to_bio(chunk, engine='newmm'):
    """
    Tokenize one chunk (no embedded spaces) and return BIO labels.
    Returns a list of (char, label) pairs for non-space characters only.
    The tokenizer may insert/remove spaces; we align back to the original
    chunk chars by matching characters directly.
    """
    # Tokenize with keep_whitespace=False so we only get word tokens
    words = word_tokenize(chunk.replace(' ', ''), engine=engine, keep_whitespace=False)
    
    # Build label list for the reconstructed non-ws chars
    pairs = []
    for word in words:
        labels = words_to_bio(word)
        for c, l in zip(word, labels):
            pairs.append((c, l))
    return pairs


def align_labels_to_original(original_non_ws, tokenizer_pairs):
    """
    Align tokenizer output (which may differ) back to original character sequence.
    Strategy: walk both sequences; when chars match → use tokenizer label.
    When tokenizer char doesn't match → mark with None (resolved by fallback).
    """
    aligned = []
    tok_idx = 0
    n_tok = len(tokenizer_pairs)
    
    for orig_char in original_non_ws:
        if tok_idx < n_tok and tokenizer_pairs[tok_idx][0] == orig_char:
            aligned.append(tokenizer_pairs[tok_idx][1])
            tok_idx += 1
        else:
            # Skip mismatched tokenizer chars (extra/modified chars)
            while tok_idx < n_tok and tokenizer_pairs[tok_idx][0] != orig_char:
                tok_idx += 1
            if tok_idx < n_tok:
                aligned.append(tokenizer_pairs[tok_idx][1])
                tok_idx += 1
            else:
                aligned.append(None)  # fallback: will use other engine
    
    return aligned


def run_tokenizer_on_text(text, non_ws_chars_full, engine='newmm'):
    """
    Run tokenizer on full text by splitting at whitespace segments.
    Returns aligned BIO labels for non_ws_chars_full.
    """
    labels_all = []
    
    # Process segment by segment (split at whitespace)
    # Each segment is a sequence of non-ws chars; spaces act as word boundaries
    segments = []
    current_seg = []
    current_seg_chars = []

    for c in text:
        if c == ' ':
            if current_seg:
                segments.append((''.join(current_seg), current_seg_chars[:]))
                current_seg = []
                current_seg_chars = []
        else:
            current_seg.append(c)
            current_seg_chars.append(c)
    if current_seg:
        segments.append((''.join(current_seg), current_seg_chars[:]))

    for seg_text, seg_chars in segments:
        if not seg_chars:
            continue
        try:
            tok_pairs = tokenize_chunk_to_bio(seg_text, engine=engine)
            seg_labels = align_labels_to_original(seg_chars, tok_pairs)
        except Exception:
            seg_labels = [None] * len(seg_chars)
        labels_all.extend(seg_labels)

    return labels_all


# =============================================================================
# 3. RUN TOKENIZERS
# =============================================================================
print("\n" + "="*60)
print("2. TOKENIZING TEST TEXT (chunk-based)")
print("="*60)

print('Running newmm...')
newmm_labels = run_tokenizer_on_text(test_text, non_ws_chars, engine='newmm')
print(f'  Got {len(newmm_labels):,} labels, {sum(1 for l in newmm_labels if l is None):,} unresolved')

print('Running longest...')
longest_labels = run_tokenizer_on_text(test_text, non_ws_chars, engine='longest')
print(f'  Got {len(longest_labels):,} labels, {sum(1 for l in longest_labels if l is None):,} unresolved')

assert len(newmm_labels) == len(non_ws_chars)
assert len(longest_labels) == len(non_ws_chars)
print('✅ Label lengths match')

# Fill None values: prefer the other engine, else default to B_WORD
final_tok_labels = []
for l1, l2 in zip(newmm_labels, longest_labels):
    if l1 is not None:
        final_tok_labels.append(l1)
    elif l2 is not None:
        final_tok_labels.append(l2)
    else:
        final_tok_labels.append('B_WORD')

# Compute ensemble soft probabilities
LABEL2INT = {'B_WORD': 0, 'I_WORD': 1, 'E_WORD': 2}
INT2LABEL  = {v: k for k, v in LABEL2INT.items()}

tok_probs = np.zeros((len(non_ws_chars), 3), dtype=np.float32)
for i, (l1, l2) in enumerate(zip(newmm_labels, longest_labels)):
    w1 = 0.5 if l1 is not None else 0.0
    w2 = 0.5 if l2 is not None else 0.0
    total = w1 + w2 if (w1 + w2) > 0 else 1.0
    if l1: tok_probs[i, LABEL2INT[l1]] += w1 / total
    if l2: tok_probs[i, LABEL2INT[l2]] += w2 / total
    if not l1 and not l2: tok_probs[i, 0] = 1.0  # default B_WORD

print('\nnewmm label dist:')
newmm_resolved = [l for l in newmm_labels if l is not None]
for k, v in Counter(newmm_resolved).items():
    print(f'  {k}: {v:,} ({v/len(newmm_resolved)*100:.1f}%)')

engine_agree = sum(1 for a, b in zip(newmm_labels, longest_labels)
                   if a is not None and b is not None and a == b)
total_both = sum(1 for a, b in zip(newmm_labels, longest_labels)
                 if a is not None and b is not None)
print(f'\nEngine agreement (where both resolved): {engine_agree/total_both*100:.1f}%')

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================
print("\n" + "="*60)
print("3. FEATURE ENGINEERING")
print("="*60)

THAI_CONSONANTS  = set('กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ')
THAI_LEAD_VOWELS = set('เแโใไ')
THAI_FOLL_VOWELS = set('ะัาิีึืุูํ')
THAI_TONEMARKS   = set('่้๊๋')
THAI_ABOVE_BELOW = set('็ิีึืุูํ์ๆ')
THAI_DIGITS      = set('๐๑๒๓๔๕๖๗๘๙')
ALL_THAI         = THAI_CONSONANTS | THAI_LEAD_VOWELS | THAI_FOLL_VOWELS | THAI_TONEMARKS | THAI_DIGITS

WINDOW = 5

def get_char_type(c):
    if c in THAI_CONSONANTS:  return 0
    if c in THAI_LEAD_VOWELS: return 1
    if c in THAI_FOLL_VOWELS: return 2
    if c in THAI_TONEMARKS:   return 3
    if c in THAI_ABOVE_BELOW: return 4
    if c in THAI_DIGITS:      return 5
    if c.isdigit():           return 6
    if c.isalpha():           return 7
    return 9


def extract_features(chars):
    """Feature matrix of shape (n_chars, n_features)."""
    n = len(chars)
    types  = [get_char_type(c) for c in chars]
    is_th  = [int(is_thai_char(c)) for c in chars]
    is_con = [int(c in THAI_CONSONANTS) for c in chars]
    is_ton = [int(c in THAI_TONEMARKS) for c in chars]
    is_lv  = [int(c in THAI_LEAD_VOWELS) for c in chars]
    is_fv  = [int(c in THAI_FOLL_VOWELS) for c in chars]
    is_dig = [int(c.isdigit() or c in THAI_DIGITS) for c in chars]
    is_alp = [int(c.isalpha()) for c in chars]
    is_pun = [int(not c.isalnum() and c not in ALL_THAI) for c in chars]

    rows = []
    for i in range(n):
        row = []
        for offset in range(-WINDOW, WINDOW + 1):
            j = i + offset
            pad = j < 0 or j >= n
            row += [
                types[j]  if not pad else -1,
                is_th[j]  if not pad else 0,
                is_con[j] if not pad else 0,
                is_ton[j] if not pad else 0,
                is_lv[j]  if not pad else 0,
                is_fv[j]  if not pad else 0,
                is_dig[j] if not pad else 0,
                is_alp[j] if not pad else 0,
                is_pun[j] if not pad else 0,
            ]
        prev_type = types[i - 1] if i > 0 else -1
        next_type = types[i + 1] if i < n - 1 else -1
        row.append(prev_type * 11 + types[i])
        row.append(types[i] * 11 + next_type)
        row.append(int(i < n - 1 and chars[i] in THAI_LEAD_VOWELS and chars[i + 1] in THAI_CONSONANTS))
        row.append(int(i > 0 and chars[i - 1] in THAI_TONEMARKS))
        row.append(int(i > 0 and is_thai_char(chars[i]) != is_thai_char(chars[i - 1])))
        rows.append(row)
    return np.array(rows, dtype=np.float32)


print('Extracting features...')
X = extract_features(non_ws_chars)
y = np.array([LABEL2INT[l] for l in final_tok_labels], dtype=np.int32)
print(f'Feature matrix: {X.shape}')
print(f'Classes: B={np.sum(y==0):,}  I={np.sum(y==1):,}  E={np.sum(y==2):,}')

# =============================================================================
# 5. TRAIN LIGHTGBM
# =============================================================================
print("\n" + "="*60)
print("4. TRAINING LIGHTGBM")
print("="*60)

model = lgb.LGBMClassifier(
    objective='multiclass', num_class=3, metric='multi_logloss',
    learning_rate=0.08, num_leaves=127, min_child_samples=20,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    n_estimators=400, class_weight='balanced',
    n_jobs=-1, verbose=-1, random_state=42,
)
model.fit(
    X, y,
    eval_set=[(X, y)],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(100)]
)
print(f'✅ Done. Best iteration: {model.best_iteration_}')

y_pred = model.predict(X)
print(f'Training Macro F1 (in-sample): {f1_score(y, y_pred, average="macro"):.4f}')
print(classification_report(y, y_pred, target_names=['B_WORD', 'I_WORD', 'E_WORD']))

# =============================================================================
# 6. ENSEMBLE PREDICTION
# =============================================================================
print("\n" + "="*60)
print("5. ENSEMBLE PREDICTION")
print("="*60)

lgb_probs = model.predict_proba(X)

# Weights: tokenizer ensemble drives most of the signal; LightGBM refines
W_TOK, W_LGB = 0.65, 0.35
ens_probs = W_TOK * tok_probs + W_LGB * lgb_probs
raw_labels = [INT2LABEL[p] for p in np.argmax(ens_probs, axis=1)]

print('Ensemble label dist:')
for k, v in Counter(raw_labels).items():
    print(f'  {k}: {v:,} ({v/len(raw_labels)*100:.1f}%)')

# =============================================================================
# 7. BIO POST-PROCESSING
# =============================================================================

def fix_bio_sequence(labels):
    """Fix invalid BIO transitions (I/E after E → change to B)."""
    fixed = list(labels)
    for i in range(len(fixed)):
        if i == 0:
            if fixed[i] != 'B_WORD':
                fixed[i] = 'B_WORD'
            continue
        prev = fixed[i - 1]
        if fixed[i] in ('I_WORD', 'E_WORD') and prev not in ('B_WORD', 'I_WORD'):
            fixed[i] = 'B_WORD'
    return fixed


final_labels = fix_bio_sequence(raw_labels)

print('\nFinal label dist:')
for k, v in Counter(final_labels).items():
    print(f'  {k}: {v:,} ({v/len(final_labels)*100:.1f}%)')

# =============================================================================
# 8. VALIDATE & SAVE
# =============================================================================
print("\n" + "="*60)
print("6. VALIDATION & SUBMISSION")
print("="*60)

assert len(final_labels) == 35182, f'Expected 35182 rows, got {len(final_labels)}'
assert all(l in ('B_WORD', 'I_WORD', 'E_WORD') for l in final_labels)
print('✅ All validation checks passed')

sub = pd.DataFrame({'Id': non_ws_ids, 'Predicted': final_labels})
sub.to_csv(OUTPUT_FILE, index=False)

print(f'\n✅ Submission saved: {OUTPUT_FILE}')
print(f'   Rows: {len(sub):,}')
print('\nFirst 15 rows:')
print(sub.head(15).to_string(index=False))
print('\nLabel distribution:')
print(sub['Predicted'].value_counts())
print("\n" + "="*60 + "\nDONE!\n" + "="*60)
