"""One version-pinned frontend shared by training and evaluation."""
import json
import re
import unicodedata

from training.common import ROOT


class Frontend:
    def __init__(self):
        from misaki import en, espeak, zh
        self.en = en.G2P(trf=False, british=False,
                         fallback=espeak.EspeakFallback(british=False), unk='❓')
        self.zh = zh.ZHG2P(version=None, unk='❓')
        self.vocab = json.loads((ROOT / 'models/Kokoro-82M/config.json').read_text())['vocab']

    def __call__(self, text, language):
        normalized = unicodedata.normalize('NFKC', text).strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        # Explicit punctuation equivalences; never remove phonetic content.
        normalized = normalized.replace('·', ' ').replace('–', '—')
        if re.search(r'[À-ÖØ-öø-ÿ]', normalized):
            raise ValueError('unsupported_nonEnglish_Latin_from_dataset')
        if language == 'en':
            phonemes = self.en(normalized)[0]
        else:
            # Misaki v1.0's legacy Chinese frontend leaves Latin words raw,
            # even when en_callable is supplied. Explicitly route those spans.
            spans = re.split(r"([A-Za-z][A-Za-z0-9_'’\-]*(?:[ \t]+[A-Za-z][A-Za-z0-9_'’\-]*)*)", normalized)
            result = []
            for span in spans:
                if not span.strip():
                    continue
                if re.match('[A-Za-z]', span):
                    result.append(self.en(span)[0])
                else:
                    result.append(self.zh(span)[0])
            phonemes = ' '.join(result)
        phonemes = re.sub(r'\s+', ' ', phonemes).strip()
        unknown = sorted(set(phonemes) - set(self.vocab))
        if unknown:
            raise ValueError(f'Unknown phoneme symbols: {unknown!r}')
        ids = [self.vocab[p] for p in phonemes]
        if not ids:
            raise ValueError('Empty phoneme sequence')
        return phonemes, ids
