const $ = (id) => document.getElementById(id);
const samples = {
  zh: '每一种声音，都有自己的温度。让我们从这段文字开始，听见语言的节奏，也听见故事里的细节。',
  en: 'Take a quiet moment to look around. Sometimes, the most beautiful stories begin with the smallest details.',
  mixed: '今天的主题是 slow living。放慢脚步，给自己一点时间，enjoy the moment，让生活重新回到喜欢的节奏。',
};
let ready = false, busy = false, audioURL = null;
const update = () => {
  $('count').textContent = $('sentence-chunking').checked ? `${$('text').value.length} 字` : `${$('text').value.length} / 600`;
  $('sentence-chunking').disabled = busy;
  $('chunk-hint').textContent = $('sentence-chunking').checked ? '按句末标点和换行分段，支持长文。关闭可对比原来的整段朗读。' : '整段朗读，保留原来的600字和单段发音长度限制。';
  $('generate').disabled = !ready || busy || !$('text').value.trim();
};
$('text').addEventListener('input', update);
$('sentence-chunking').addEventListener('change', update);
$('speed').addEventListener('input', () => $('speed-value').textContent = `${Number($('speed').value).toFixed(2)}×`);
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => {
  $('text').value = samples[button.dataset.example]; $('language').value = 'auto'; update();
}));
async function connect() {
  try {
    const response = await fetch('/healthz', {cache: 'no-store'});
    if (!response.ok) throw new Error();
    ready = (await response.json()).status === 'ready';
  } catch { ready = false; }
  $('service-status').textContent = ready ? '服务已就绪' : '等待服务连接';
  document.querySelector('.status').classList.toggle('ready', ready); update();
  if (!ready) setTimeout(connect, 4000);
}
async function generate() {
  if ($('generate').disabled) return;
  busy = true; update(); $('error').hidden = true; $('button-text').textContent = '正在生成…';
  document.querySelector('.listen').classList.add('busy');
  $('announcement').textContent = '正在生成语音，请稍候。';
  if (!audioURL) { $('empty-text').textContent = '正在为文字注入声音'; $('empty-note').textContent = '首次合成可能需要稍长时间，请稍候。'; }
  $('audio').pause();
  try {
    const response = await fetch('/api/synthesize', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text: $('text').value, language: $('language').value, speed: Number($('speed').value), sentence_chunking: $('sentence-chunking').checked})});
    if (!response.ok) {
      const data = await response.json();
      throw new Error(typeof data.detail === 'string' ? data.detail : '请检查文字长度和朗读设置后重试。');
    }
    const blob = await response.blob();
    if (audioURL) URL.revokeObjectURL(audioURL);
    audioURL = URL.createObjectURL(blob); $('audio').src = audioURL; $('download').href = audioURL;
    $('duration').textContent = `音频 ${Number(response.headers.get('X-Audio-Duration')).toFixed(1)} 秒`;
    $('elapsed').textContent = `合成用时 ${Number(response.headers.get('X-Synthesis-Seconds')).toFixed(1)} 秒`;
    $('result-label').textContent = response.headers.get('X-Sentence-Chunking') === 'true' ? `每句一块 · ${response.headers.get('X-Chunk-Count')} 段` : '整段朗读';
    $('empty').hidden = true; $('result').hidden = false; $('announcement').textContent = '语音已生成，可以播放或下载。';
  } catch (error) {
    $('error').textContent = error instanceof TypeError ? '连接中断，请检查网络后重试。' : error.message;
    $('error').hidden = false; $('announcement').textContent = '生成未完成。';
    $('empty-text').textContent = '声音，即将在这里出现'; $('empty-note').textContent = '调整文字后可以重新生成。';
  } finally {
    busy = false; $('button-text').textContent = '生成语音'; document.querySelector('.listen').classList.remove('busy'); update();
  }
}
$('generate').addEventListener('click', generate);
$('text').addEventListener('keydown', event => {if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {event.preventDefault(); generate();}});
window.addEventListener('beforeunload', () => {if (audioURL) URL.revokeObjectURL(audioURL);});
update(); connect();
