"""Conservative sentence/content checks; no splitting or concatenation of utterances."""
import re
import unicodedata


def classify(text):
    han=len(re.findall(r'[\u4e00-\u9fff]',text))
    words=len(re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?",text))
    return ('mixed' if han and words else 'zh' if han else 'en'),han,words


def sentences(text):
    text=re.sub(r'\.{2,}','…',text)
    text=re.sub(r'\b(?:[A-Za-z]\.){2,}',lambda m:m[0].replace('.','∯'),text)
    protected=re.sub(r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc)\.',lambda m:m[0][:-1]+'∯',text,flags=re.I)
    protected=re.sub(r'(?<=\d)\.(?=\d)','∯',protected)
    protected=re.sub(r'\b[A-Za-z]\.(?=\s*[A-Za-z]\.)',lambda m:m[0][:-1]+'∯',protected)
    parts=[];start=0
    for match in re.finditer(r'[。！？!?]+|\.(?=\s|[”"\u2019\']|$)',protected):
        piece=protected[start:match.end()].strip(' \t\n“”"\u2018\u2019');start=match.end()
        _,han,words=classify(piece)
        # Initials, abbreviations, and one-word exclamations are not complete sentences.
        if han>=6 or words>=3 or han+3*words>=9:parts.append(piece.replace('∯','.'))
    tail=protected[start:].strip(' \t\n“”"\u2018\u2019')
    if tail:return []
    return parts


def check(text):
    if not isinstance(text,str):return None,'not_text'
    text=unicodedata.normalize('NFKC',text).strip()
    language,han,words=classify(text)
    if not text or len(text)>1400 or re.search(r'https?://|www\.|@|[<>\[\]{}|]',text):return None,'format'
    if re.search(r'[À-ÖØ-öø-ÿ]',text):return None,'unsupported_latin'
    parts=sentences(text)
    if len(parts)<3:return None,'fewer_than_3_sentences'
    if language=='mixed' and (han<10 or words<3):return None,'mixed_content'
    estimate=han/4.5+words/2.5
    if not 17<=estimate<=45:return None,'estimated_duration'
    if re.search(r'(.{4,12})\1\1',re.sub(r'\W','',text)):return None,'repetition'
    return dict(text=text,language=language,sentence_count=len(parts),estimated_seconds=estimate),None
