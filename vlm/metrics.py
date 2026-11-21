"""Caption metrics: corpus BLEU-4 and CIDEr-D, following the coco-caption reference code.

Pure Python, no Java. Tokenization is lowercase + strip punctuation, close to the PTB
tokenizer used by coco-caption, so scores are comparable in spirit but not to the decimal.
"""
import math
import re
from collections import Counter, defaultdict

_PUNCT = re.compile(r"[^a-z0-9' ]+")


def tokenize(s):
    return _PUNCT.sub(" ", s.lower()).split()


def _ngrams(words, n):
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def bleu(hyps, refs, max_n=4):
    """Corpus BLEU-1..4. hyps: [str], refs: [[str]]. Returns list of 4 floats."""
    match, total = [0] * max_n, [0] * max_n
    hyp_len = ref_len = 0
    for h, rs in zip(hyps, refs):
        hw, rws = tokenize(h), [tokenize(r) for r in rs]
        hyp_len += len(hw)
        # "closest" reference length, ties to the shorter one (coco-caption behaviour)
        ref_len += min((abs(len(r) - len(hw)), len(r)) for r in rws)[1]
        for n in range(1, max_n + 1):
            hc = _ngrams(hw, n)
            maxref = Counter()
            for r in rws:
                maxref |= _ngrams(r, n)
            match[n - 1] += sum(min(c, maxref[g]) for g, c in hc.items())
            total[n - 1] += max(len(hw) - n + 1, 0)
    tiny = 1e-9
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    out, logsum = [], 0.0
    for n in range(max_n):
        logsum += math.log((match[n] + tiny) / (total[n] + tiny))
        out.append(bp * math.exp(logsum / (n + 1)))
    return out


def cider_d(hyps, refs, n=4, sigma=6.0):
    """CIDEr-D (x10 scale, as reported in papers). Document frequency from the references."""
    ref_toks = [[tokenize(r) for r in rs] for rs in refs]
    df = defaultdict(float)
    for rs in ref_toks:
        for g in set(g for r in rs for k in range(1, n + 1) for g in _ngrams(r, k)):
            df[g] += 1
    log_n = math.log(float(len(refs)))

    def vec(words):
        vs, norms = [], []
        for k in range(1, n + 1):
            v = {g: c * (log_n - math.log(max(1.0, df[g]))) for g, c in _ngrams(words, k).items()}
            vs.append(v)
            norms.append(math.sqrt(sum(x * x for x in v.values())))
        return vs, norms, len(words)

    def sim(hv, hn, hl, rv, rn, rl):
        out = []
        for k in range(n):
            val = sum(min(hv[k][g], rv[k].get(g, 0.0)) * rv[k].get(g, 0.0) for g in hv[k])  # clipped
            if hn[k] and rn[k]:
                val /= hn[k] * rn[k]
            else:
                val = 0.0
            out.append(val * math.exp(-((hl - rl) ** 2) / (2 * sigma ** 2)))
        return out

    scores = []
    for h, rs in zip(hyps, ref_toks):
        hv, hn, hl = vec(tokenize(h))
        acc = [0.0] * n
        for r in rs:
            for k, s in enumerate(sim(hv, hn, hl, *vec(r))):
                acc[k] += s
        scores.append(sum(acc) / n / len(rs) * 10.0)
    return sum(scores) / len(scores), scores
