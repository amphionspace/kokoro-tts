"""Bounded regression checks for sentence boundaries and request validation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from demo.server import split_sentences,SynthesisRequest,Engine
cases={
 '你好，欢迎来到这里。':['你好，欢迎来到这里。'],
 'Hello, world.':['Hello, world.'],
 '今天聊聊 AI，then we try it.':['今天聊聊 AI，then we try it.'],
 '你好。再见！下一句？':['你好。','再见！','下一句？'],
 '她说：“你好！”然后走了。':['她说：“你好！”','然后走了。'],
 'Dr. Smith paid 3.14 dollars. Next sentence!':['Dr. Smith paid 3.14 dollars.','Next sentence!'],
 'Visit example.com today.\nNew line':['Visit example.com today.','New line'],
 'Mr. Jones met A. Smith. Hello.':['Mr. Jones met A. Smith.','Hello.'],
 'English sentence.中文句子。':['English sentence.','中文句子。'],
 '你好\n\nworld':['你好','world'],
 'Wait... Really?!':['Wait...','Really?!'],
}
for text,expected in cases.items():
 actual=split_sentences(text)
 assert actual==expected,(text,actual,expected)
 assert ''.join(actual).replace(' ','')==''.join(text.split()),(text,actual)
assert SynthesisRequest(text='你好。'*300).sentence_chunking
class Dummy:
 def __init__(self):self.calls=0
 def forward_with_tokens(self,*args,**kwargs):self.calls+=1;raise AssertionError('must validate before synthesis')
engine=Engine.__new__(Engine);engine.model=Dummy();engine.frontend=lambda s,l:('',[1]*(511 if '长' in s else 10))
try:engine.synthesize(SynthesisRequest(text='你好。长句。'))
except ValueError as e:assert '第2段' in str(e) and '511' in str(e)
else:raise AssertionError('expected length error')
assert engine.model.calls==0
try:engine.synthesize(SynthesisRequest(text='你好。'*300,sentence_chunking=False))
except ValueError as e:assert '600' in str(e)
else:raise AssertionError('expected whole-text limit')
print('11 segmentation cases, no content dropped, later-sentence prevalidation, and whole-text limit passed.')
