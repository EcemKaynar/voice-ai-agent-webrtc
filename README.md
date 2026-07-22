# Voice AI Agent WebRTC

Bu proje, WebRTC tabanlı Türkçe bir sesli yapay zekâ asistanı MVP’sidir.

Tarayıcıdan mikrofon sesini alır, kullanıcının konuşmasını otomatik algılar, sesi yazıya çevirir, doküman tabanlı bilgi tabanında topic-aware semantic RAG ile arama yapar, LLM ile bilgiye dayalı cevap üretir ve cevabı Türkçe ses olarak kullanıcıya döndürür.

Bu sürüm, doküman tabanlı Türkçe müşteri destek senaryolarına odaklanır. Mevcut uygulamada bilgi tabanı Garenta araç kiralama süreçleri için kullanılmıştır.

---

## Özellikler

- Tarayıcı üzerinden WebRTC ile mikrofon girişi
- RMS ve sessizlik algılama ile otomatik konuşma tespiti
- Kullanıcı konuşmasının WAV ses dosyası olarak kaydedilmesi
- faster-whisper ile Türkçe STT
- Qdrant local mode ile doküman tabanlı semantic RAG
- Daha doğru bağlam seçimi için topic-aware retrieval
- Tablo benzeri bilgiler için structured chunk üretimi
- Bilgi tabanına bağlı Türkçe prompt servisi
- OpenRouter ile LLM cevabı üretimi
- Streaming LLM çağrısı ve başarısız olursa non-streaming fallback
- Eksik, boş, İngilizce/meta veya güvenli olmayan LLM cevaplarını algılama
- LLM kullanılamadığında extractive RAG fallback
- Edge TTS ile Türkçe sesli cevap üretimi
- Modern tarayıcı arayüzü
- SQLite tabanlı sesli işlem metrikleri
- RAG ve LLM davranışını test etmek için text-test endpoint’i

---

## Mimari

```text
Tarayıcı mikrofonu
        ↓
WebRTC ses akışı
        ↓
Otomatik konuşma algılama
        ↓
WAV ses parçası
        ↓
STT / faster-whisper
        ↓
Transcript normalization
        ↓
Topic-aware semantic RAG
        ↓
Prompt service
        ↓
LLM / OpenRouter
        ↓
Gerekirse extractive RAG fallback
        ↓
TTS / Edge TTS
        ↓
Tarayıcıda ses oynatma
        ↓
Metrik kaydı
```

---

## Nasıl Çalışır?

Sistemde iki ana kullanım modu vardır:

```text
1. Sesli mod
   Tarayıcı mikrofonu → STT → RAG → LLM/fallback → TTS → tarayıcıda oynatma

2. Yazılı test modu
   Query parametresi → RAG → LLM/fallback → JSON cevap
```

Yazılı test modu, mikrofon kullanmadan bilgi tabanı aramasını ve cevap üretimini kontrol etmek için kullanılır.

---

## RAG Tasarımı

Orijinal bilgi tabanı dokümanı değiştirilmez.

Sistem dokümanı okur, parçalara böler, gerekli yerlerde structured chunk üretir, her parçaya topic metadata ekler ve bunları Qdrant içine indexler.

Mevcut RAG akışı:

```text
Orijinal doküman
        ↓
Doküman parser
        ↓
Chunking
        ↓
Structured chunk üretimi
        ↓
Topic metadata
        ↓
Vector embedding
        ↓
Qdrant local index
        ↓
Topic-aware semantic retrieval
        ↓
LLM cevabı veya extractive fallback
```

---

## Topic-Aware Retrieval

Sadece semantic search kullanıldığında bazı benzer ifadeler karışabilir:

```text
ek sürücü     vs genç sürücü
geç teslim   vs yakıt teslimi
ödeme         vs rezervasyon/iade metinleri
```

Bu karışıklığı azaltmak için projede topic routing katmanı kullanılır.

Mevcut topic’ler:

```text
additional_driver
late_return
payment
fuel
accident_damage
segment_conditions
cancellation_return
```

Sistem önce kullanıcı sorusunun hangi konuya ait olduğunu tahmin eder. Daha sonra semantic benzerlik ve topic metadata birlikte kullanılarak sonuçlar yeniden sıralanır.

Bu yöntem final cevabı kod içine hardcode etmeden arama doğruluğunu artırır.

---

## Önemli Servisler

### `semantic_knowledge_base_service.py`

Bu servis şunlardan sorumludur:

- `.docx`, `.txt` ve `.md` bilgi tabanı dosyalarını okuma
- Metin chunk’ları oluşturma
- Tablo benzeri bilgilerden structured segment chunk üretme
- Chunk’ları Qdrant içine indexleme
- Semantic search çalıştırma
- Lexical ve topic-aware reranking uygulama
- RAG context oluşturma
- LLM başarısız olduğunda extractive fallback cevap üretme

---

### `topic_router_service.py`

Bu servis şunlardan sorumludur:

- Kullanıcı sorusunun olası topic’ini bulma
- Chunk’lara topic metadata atama
- Benzer kavramların karışmasını azaltma
- Topic uyumuna göre chunk filtreleme veya yeniden sıralama

Örnek:

```text
"ek sürücü ekleyebilir miyim?"
→ topic: additional_driver

"genç sürücü ne koşulda oluyor?"
→ topic: segment_conditions

"aracı geç teslim edersem ne olur?"
→ topic: late_return
```

---

### `prompt_service.py`

Bu servis, bilgi tabanına bağlı Türkçe prompt oluşturur.

Prompt, LLM’e şunları söyler:

- Sadece verilen bilgi tabanına göre cevap ver
- Desteklenmeyen çıkarım yapma
- Alakasız konuları karıştırma
- Sayı, limit, ücret, tarih, yaş ve koşul bilgisi varsa cevaba dahil et
- Kısa, doğal ve Türkçe cevap üret

---

### `llm_service.py`

Bu servis şunlardan sorumludur:

- OpenRouter çağrısı yapma
- Önce streaming cevap deneme
- Streaming başarısız olursa non-streaming deneme
- Model çıktısını temizleme
- Eksik cevapları reddetme
- İngilizce, meta veya prompt benzeri cevapları reddetme
- LLM kullanılamazsa local fallback çıktısı döndürme

---

### `transcript_normalizer_service.py`

Bu servis şu anda transcript’i değiştirmeden geri döndürür.

Bunun nedeni, yanlış domain düzeltmelerinin önüne geçmektir. Örneğin sistemin bir Garenta terimini yanlışlıkla başka bir konuya çevirmesi engellenir.

---

## Extractive RAG Fallback

Extractive fallback şu durumlarda kullanılır:

- OpenRouter rate limit yediğinde
- LLM boş cevap döndürdüğünde
- LLM eksik cevap döndürdüğünde
- LLM meta/prompt benzeri metin döndürdüğünde
- LLM güvenli şekilde bilgiye dayalı cevap üretemediğinde

Bu fallback katmanı final cevapları hardcode etmez.

Bunun yerine, RAG sonucunda bulunan doküman parçalarından ilgili cümleleri veya structured chunk’ları seçer ve kullanıcıya uygun Türkçe bir cevap haline getirir.

---

## Örnek Sorular

Sistem şu tarz sorulara cevap verebilir:

```text
Ek sürücü ekleyebilir miyim?
Aracı geç teslim edersem ne olur?
Ödeme için banka kartı kullanabilir miyim?
Kaza yaparsam ne yapmalıyım?
Genç sürücü ne koşulda oluyor?
Lüks segmentin şartları nelerdir?
Eksik yakıtla teslim edersem ne olur?
Yardım hattı numarası nedir?
```

---

## Örnek Cevaplar

### Soru

```text
Lüks segmentin şartları nelerdir?
```

### Olası cevap

```text
Lüks segment yaş ve ehliyet koşulları: minimum sürücü yaşı 27, genç sürücü yaşı 25, minimum ehliyet yılı 5, genç sürücü ehliyet yılı 3. Lüks segment Findeks koşulu 1.400 olarak belirtilmiştir.
```

---

### Soru

```text
Ödeme için banka kartı kullanabilir miyim?
```

### Olası cevap

```text
Banka kartı ve sanal kart geçerli değildir. Ödeme ve teminat işlemleri şahsi kredi kartı üzerinden yapılmalıdır.
```

---

### Soru

```text
Ek sürücü ekleyebilir miyim?
```

### Olası cevap

```text
Kiraladığınız aracı yalnızca sözleşme ve teslimat formunda belirtilen kişiler kullanabilir. Araç kullanıcı sayısını arttırabilmek için Ek Sürücü hizmeti satın alabilirsiniz. Bir araç için en fazla 5 adet ek sürücü tanımlanabilir.
```

---

### Soru

```text
Aracı geç teslim edersem ne olur?
```

### Olası cevap

```text
Olası gecikmeler durumunda 2 saat üzeri gecikmede günlük kira bedelinin 1/3’ü, 3 saat ve üzeri gecikmede 2/3’ü, 4 saat ve üzeri gecikmede ise bir günlük kira bedeli uygulanır. Teslim ya da iade saatiniz ofis kapanış saatindeyse 2 saate kadar gecikme opsiyonu geçerli değildir.
```

---

## Kullanılan Teknolojiler

- Python
- FastAPI
- aiortc
- faster-whisper
- Qdrant local mode
- sentence-transformers
- OpenRouter API
- Edge TTS
- SQLite
- HTML / CSS / JavaScript

---

## Proje Yapısı

```text
voice-ai-agent-webrtc/
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── web/
│   └── index.html
├── services/
│   ├── __init__.py
│   ├── stt_service.py
│   ├── tts_service.py
│   ├── llm_service.py
│   ├── prompt_service.py
│   ├── semantic_knowledge_base_service.py
│   ├── topic_router_service.py
│   └── transcript_normalizer_service.py
├── utils/
│   ├── audio_utils.py
│   └── metrics.py
├── knowledge_base/
│   └── knowledge_base_document.docx
└── data/
    ├── audio_chunks/
    ├── tts_outputs/
    ├── qdrant/
    └── voice_metrics.db
```

Aşağıdaki dosya ve klasörler GitHub’a yüklenmemelidir:

```text
data/
.env
local database dosyaları
üretilen ses dosyaları
Qdrant index dosyaları
Python cache dosyaları
```

---

## Kurulum

### 1. Virtual environment oluştur

```bash
python -m venv .venv
```

Windows PowerShell’de aktif et:

```powershell
.venv\Scripts\Activate.ps1
```

---

### 2. Bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

Önemli paketler:

```text
fastapi
uvicorn
aiortc
python-dotenv
requests
edge-tts
faster-whisper
qdrant-client
sentence-transformers
```

---

### 3. `.env` dosyası oluştur

Proje kök dizininde `.env` dosyası oluştur.

Örnek:

```env
APP_NAME=Voice AI Agent WebRTC

OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openrouter/free
LLM_TIMEOUT_SECONDS=30

SYSTEM_PROMPT=Sen Türkçe konuşan kısa ve doğal cevap veren bir sesli AI asistansın.

STT_PROVIDER=faster_whisper
WHISPER_MODEL_SIZE=small
WHISPER_LANGUAGE=tr
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_BEAM_SIZE=5
WHISPER_BEST_OF=5
WHISPER_VAD_FILTER=true
WHISPER_CONDITION_ON_PREVIOUS_TEXT=false
WHISPER_INITIAL_PROMPT=Garenta araç kiralama süreçleri. Ek sürücü, genç sürücü, geç teslim, geç iade, yakıt, teminat, kredi kartı, ödeme, iptal, erken iade, no show, hasar, kaza, güvence, kilometre limiti.

TTS_PROVIDER=edge_tts
TTS_VOICE=tr-TR-EmelNeural
TTS_OUTPUT_DIR=data/tts_outputs

AUDIO_OUTPUT_DIR=data/audio_chunks
TARGET_SAMPLE_RATE=16000

SPEECH_RMS_THRESHOLD=150
MIN_AUDIO_RMS=70
SILENCE_END_SECONDS=1.2
MIN_UTTERANCE_SECONDS=1.0
MAX_UTTERANCE_SECONDS=15
PRE_SPEECH_SECONDS=0.45
MIN_SPEECH_SECONDS=0.45
MIN_SPEECH_RATIO=0.10

KNOWLEDGE_BASE_DIR=knowledge_base

QDRANT_LOCAL_PATH=data/qdrant
QDRANT_COLLECTION=garenta_knowledge_base
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RAG_CHUNK_SIZE=900
RAG_CHUNK_OVERLAP=0
RAG_SCORE_THRESHOLD=0.28
```

---

### 4. Bilgi tabanı dokümanı ekle

`.docx`, `.txt` veya `.md` formatındaki bilgi tabanı dosyasını şu klasöre koy:

```text
knowledge_base/
```

Örnek:

```text
knowledge_base/garenta_kiralama_kosullari.docx
```

Sistem bu dokümanı okuyup local Qdrant index oluşturur.

---

### 5. Sunucuyu çalıştır

```bash
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

Tarayıcı arayüzü:

```text
http://127.0.0.1:8001
```

API dokümantasyonu:

```text
http://127.0.0.1:8001/docs
```

---

## Yararlı Endpoint’ler

### Health check

```text
GET /health
```

Sunucu, STT, TTS ve bilgi tabanı durumunu döndürür.

---

### Config

```text
GET /config
```

Çalışma zamanı konfigürasyonunu döndürür.

---

### Text test

```text
GET /text-test?q=ek sürücü ekleyebilir miyim?
```

Yazılı test pipeline’ını çalıştırır:

```text
query → RAG → LLM → gerekirse fallback → JSON cevap
```

---

### LLM olmadan text test

```text
GET /text-test?q=ek sürücü ekleyebilir miyim?&use_llm=false
```

Sadece RAG ve extractive fallback katmanını çalıştırır.

OpenRouter’a bağlı kalmadan retrieval kalitesini test etmek için kullanılır.

---

### Knowledge search

```text
GET /knowledge/search?q=ödeme için banka kartı kullanabilir miyim
```

Ham RAG arama sonuçlarını döndürür.

---

### Knowledge reload

```text
POST /knowledge/reload
```

Bilgi tabanı dosyalarından Qdrant index’ini yeniden oluşturur.

---

### Latest response

```text
GET /latest-response
```

Frontend için son işlenen sesli cevabı döndürür.

---

### Metrics

```text
GET /metrics
```

Son STT / LLM / TTS pipeline metriklerini döndürür.

---

## RAG Index’i Yeniden Oluşturma

Doküman parsing, chunking, topic metadata veya structured chunk mantığı değiştiğinde local Qdrant index yeniden oluşturulmalıdır.

Windows PowerShell:

```powershell
Ctrl + C
Remove-Item -Recurse -Force data\qdrant -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force services\__pycache__ -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force __pycache__ -ErrorAction SilentlyContinue
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

Ardından Swagger üzerinden şu endpoint çalıştırılır:

```text
POST /knowledge/reload
```

Swagger adresi:

```text
http://127.0.0.1:8001/docs
```

---

## Test

### Sadece RAG ve fallback testi

```text
http://127.0.0.1:8001/text-test?q=ek sürücü ekleyebilir miyim?&use_llm=false
http://127.0.0.1:8001/text-test?q=aracı geç teslim edersem ne olur?&use_llm=false
http://127.0.0.1:8001/text-test?q=ödeme için banka kartı kullanabilir miyim&use_llm=false
http://127.0.0.1:8001/text-test?q=genç sürücü ne koşulda oluyor?&use_llm=false
```

Beklenen topic davranışı:

```text
ek sürücü      → topic: additional_driver
geç teslim    → topic: late_return
ödeme          → topic: payment
genç sürücü   → topic: segment_conditions
```

---

### LLM dahil tam pipeline testi

```text
http://127.0.0.1:8001/text-test?q=ek sürücü ekleyebilir miyim?
http://127.0.0.1:8001/text-test?q=aracı geç teslim edersem ne olur?
http://127.0.0.1:8001/text-test?q=ödeme için banka kartı kullanabilir miyim
http://127.0.0.1:8001/text-test?q=genç sürücü ne koşulda oluyor?
```

Beklenen cevap kaynağı:

```text
answer_source: natural_llm_answer
```

OpenRouter rate limit veya hata verirse şu da kabul edilebilir:

```text
answer_source: extractive_rag_fallback
```

Final cevap doğru ve dokümana dayalı olduğu sürece iki durum da geçerlidir.

---

## OpenRouter Rate Limit

OpenRouter free modelleri bazen şu hatayı döndürebilir:

```text
HTTP 429 Rate limit exceeded
```

Bu proje hatası değildir.

Bu durumda sistem otomatik olarak extractive RAG fallback’e düşer ve cevabı dokümandan üretmeye devam eder.

Daha stabil demo için ücretli veya daha güvenilir bir OpenRouter modeli kullanılmalıdır.

---

## Mevcut Sınırlamalar

Bu proje bir MVP’dir, production-ready bir sesli asistan değildir.

Bilinen sınırlamalar:

- STT gerçek zamanlı streaming transcription değildir; segment bazlı çalışır.
- WebRTC tarayıcı mikrofon girişi için kullanılır, TTS ise HTTP streaming ile döner.
- OpenRouter free modelleri rate limit’e takılabilir.
- Doküman anlama kalitesi parsing, chunking, topic routing ve retrieval kalitesine bağlıdır.
- Mevcut uygulama Türkçe doküman tabanlı destek sorularına optimize edilmiştir.
- Daha gelişmiş production sürümde daha güçlü reranker, daha iyi observability ve stabil ücretli LLM modeli kullanılmalıdır.
- Extractive fallback güvenilirlik için tasarlanmıştır; LLM erişilebilir olduğunda doğal LLM cevabı tercih edilir.

---

## GitHub Güvenlik Notları

GitHub’a yüklenmemesi gerekenler:

```text
.env
data/
*.db
*.sqlite
*.sqlite3
*.wav
*.mp3
__pycache__/
.venv/
```

Public repository için özel/şirket içi dokümanlar yüklenmemelidir.

Gerçek şirket dokümanı yerine örnek bir knowledge base dokümanı kullanılmalıdır.

---

## Önerilen `.gitignore`

```gitignore
.venv/
__pycache__/
*.pyc

.env
.env.local

data/
*.db
*.sqlite
*.sqlite3

*.wav
*.mp3

.DS_Store
```

---

## Önerilen Git Komutları

```bash
git status
git add app.py README.md requirements.txt .gitignore web/index.html services utils
git status
git commit -m "Improve semantic RAG voice agent MVP"
git push
```

Özel bilgi tabanı dosyaları, repository private değilse GitHub’a eklenmemelidir.

---

## Özet

Bu proje şu yapıyı kullanan Türkçe bir sesli müşteri destek asistanı MVP’sidir:

```text
WebRTC + STT + topic-aware semantic RAG + LLM + extractive fallback + TTS streaming
```

Sistem, dokümana dayalı süreç sorularını cevaplayabilir ve LLM sağlayıcısı kullanılamadığında veya rate limit’e takıldığında bile dokümandan cevap üretmeye devam eder.