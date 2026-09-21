"""Single-process web demo for the exported Majestic Kokoro model."""
import argparse
import asyncio
from contextlib import asynccontextmanager
import io
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STATIC = Path(__file__).parent / 'static'
DEFAULT_EXPORT = ROOT / 'runs/majestic_v1_20260920/eval/stage2_final'
logger = logging.getLogger('kokoro.demo')


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1)
    language: Literal['auto', 'zh', 'en', 'mixed'] = 'auto'
    speed: float = Field(default=1.0, ge=0.7, le=1.3)
    sentence_chunking: bool = True


def split_sentences(text: str) -> list[str]:
    """Preserve punctuation and closing quotes; avoid common English false boundaries."""
    chunks, start = [], 0
    for match in re.finditer(r'[。！？!?]+|\.+|…+|\n+', text):
        if match.start() < start:
            continue
        end = match.end()
        if match.group() == '.':
            before, after = text[:end], text[end:]
            # Decimals, domains and dotted abbreviations are not sentence ends.
            if match.start() and after and re.match(r'[A-Za-z0-9]', text[match.start()-1]) and re.match(r'[A-Za-z0-9]', after[0]):
                continue
            if re.search(r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|e\.g|i\.e)\.$', before, re.I):
                continue
            if re.search(r'\b[A-Z]\.$', before) and re.match(r'\s+[A-Z][a-z]', after):
                continue
        while end < len(text) and text[end] in '\"\'”’」』】）》)':
            end += 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    if text[start:].strip():
        chunks.append(text[start:].strip())
    return chunks


class Engine:
    def __init__(self):
        import numpy as np
        import soundfile as sf
        import torch
        from training.frontend import Frontend
        from kokoro import KModel
        self.np, self.sf, self.torch = np, sf, torch
        torch.set_num_threads(4)
        self.device = os.environ.get('KOKORO_DEMO_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        folder = Path(os.environ.get('KOKORO_DEMO_EXPORT', str(DEFAULT_EXPORT))).resolve()
        self.frontend = Frontend()
        self.model = KModel(repo_id='hexgrad/Kokoro-82M',
                            config=str(ROOT / 'models/Kokoro-82M/config.json'),
                            model=str(folder / 'kokoro.pth')).to(self.device).eval()
        voice = torch.load(folder / 'majestic.pt', map_location='cpu', weights_only=True)
        if voice.shape != (510, 1, 256) or not torch.isfinite(voice).all():
            raise ValueError('Expected an exported finite [510, 1, 256] voicepack')
        self.voice = voice[0].to(self.device)

    def synthesize(self, request):
        started = time.monotonic()
        text = request.text.strip()
        if not text:
            raise ValueError('请输入要朗读的文字。')
        if not request.sentence_chunking and len(request.text) > 600:
            raise ValueError('整段模式最多600字，请开启“每句一块”或缩短文字。')
        language = request.language
        if language == 'auto':
            chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
            latin = bool(re.search(r'[A-Za-z]', text))
            language = 'mixed' if chinese and latin else ('zh' if chinese else 'en')
        segments = split_sentences(text) if request.sentence_chunking else [text]
        token_chunks = []
        # Validate every sentence before synthesis; never silently truncate a later sentence.
        for index, segment in enumerate(segments, 1):
            try:
                _, ids = self.frontend(segment, language)
            except ValueError as exc:
                raise ValueError(f'第{index}段含有暂不支持的发音符号，请检查或改写后重试。') from exc
            if len(ids) > 510:
                raise ValueError(f'第{index}段的发音长度为{len(ids)}，超过单段上限510，请补充句末标点或换行拆分。')
            token_chunks.append(ids)
        stream = io.BytesIO()
        samples = 0
        with self.sf.SoundFile(stream, mode='w', samplerate=24000, channels=1,
                               format='WAV', subtype='PCM_16') as output:
            for ids in token_chunks:
                with self.torch.inference_mode():
                    audio, _ = self.model.forward_with_tokens(
                        self.torch.tensor([[0, *ids, 0]], device=self.device), self.voice, speed=request.speed)
                    wave = audio.float().cpu().numpy().reshape(-1)
                if not self.np.isfinite(wave).all() or not len(wave):
                    raise RuntimeError('Invalid synthesis waveform')
                if not request.sentence_chunking and len(wave) > 24000 * 120:
                    raise RuntimeError('Invalid synthesis waveform')
                output.write(self.np.clip(wave, -1, 1))
                samples += len(wave)
        return stream.getvalue(), samples / 24000, time.monotonic() - started, language, len(token_chunks)


@asynccontextmanager
async def lifespan(app):
    app.state.engine = await asyncio.to_thread(Engine)
    app.state.synthesis_lock = asyncio.Lock()
    yield
    del app.state.engine


app = FastAPI(title='Kokoro · 大气女声', lifespan=lifespan, docs_url=None, redoc_url=None)
app.mount('/static', StaticFiles(directory=STATIC), name='static')


@app.get('/')
def index():
    return FileResponse(STATIC / 'index.html')


@app.get('/healthz')
def health():
    return {'status': 'ready', 'sample_rate': 24000, 'max_characters': None,
            'whole_text_max_characters': 600, 'max_tokens_per_chunk': 510,
            'sentence_chunking_default': True,
            'voice': 'majestic', 'model': 'Kokoro Stage 2 final'}


@app.post('/api/synthesize')
async def synthesize(request: SynthesisRequest):
    lock = app.state.synthesis_lock
    if lock.locked():
        raise HTTPException(429, '正在处理另一条朗读，请稍后重试。')
    async with lock:
        # Shield the worker so a disconnected browser cannot release the GPU lock early.
        task = asyncio.create_task(asyncio.to_thread(app.state.engine.synthesize, request))
        try:
            payload, duration, elapsed, language, chunks = await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            logger.exception('Speech synthesis failed')
            raise HTTPException(500, '合成失败，请稍后重试或缩短文字。') from exc
    return Response(payload, media_type='audio/wav', headers={
        'Content-Disposition': 'attachment; filename="kokoro-majestic.wav"',
        'Cache-Control': 'no-store', 'X-Audio-Duration': f'{duration:.3f}',
        'X-Synthesis-Seconds': f'{elapsed:.3f}', 'X-Language': language,
        'X-Chunk-Count': str(chunks),
        'X-Sentence-Chunking': str(request.sentence_chunking).lower(),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=32002)
    parser.add_argument('--device', default=None)
    parser.add_argument('--export-dir', type=Path, default=DEFAULT_EXPORT)
    args = parser.parse_args()
    os.environ['KOKORO_DEMO_EXPORT'] = str(args.export_dir.resolve())
    if args.device:
        os.environ['KOKORO_DEMO_DEVICE'] = args.device
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, workers=1)
