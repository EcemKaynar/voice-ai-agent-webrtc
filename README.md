# Voice AI Agent WebRTC

WebRTC tabanlı, knowledge base destekli voice AI agent MVP projesidir.

Bu proje, kullanıcının mikrofon sesini tarayıcıdan backend'e WebRTC ile aktarır. Backend tarafında konuşma algılanır, ses STT ile metne çevrilir, ilgili bilgi knowledge base üzerinden bulunur, prompt service ile LLM'e bağlamlı şekilde gönderilir ve cevap TTS ile sesli olarak kullanıcıya döndürülür.

Bu versiyonda Garenta kiralama koşulları dokümanı knowledge base olarak kullanılmıştır.

---

## Amaç

Projenin amacı, şirket süreçlerine göre cevap verebilen bir sesli AI agent prototipi oluşturmaktır.

Temel pipeline:

```text
Microphone
→ WebRTC
→ STT
→ Knowledge Base
→ Prompt Service
→ LLM
→ Streaming TTS
→ Web UI
→ Metrics
```

---

## Mevcut Özellikler

- WebRTC ile tarayıcıdan backend'e mikrofon sesi aktarımı
- Otomatik konuşma algılama
- Sessizlik algılandığında konuşma parçasını işleme
- Faster Whisper ile STT
- Whisper model preload desteği
- Mikrofon seçimi desteği
- Knowledge base doküman okuma
- DOCX tabanlı knowledge base parsing
- Keyword / intent tabanlı knowledge search
- Prompt service ile bilgi tabanına bağlı cevap üretimi
- OpenRouter üzerinden LLM cevabı üretme
- LLM streaming ve non-streaming fallback
- İngilizce/meta LLM cevaplarını filtreleme
- KB fallback cevabı
- Edge TTS ile Türkçe ses üretimi
- HTTP StreamingResponse ile TTS audio streaming
- Asistan konuşurken mikrofon input guard
- STT / LLM / TTS latency ölçümü
- SQLite metric kayıtları
- Modern web dashboard arayüzü

---

## Güncel Akış

```text
1. Kullanıcı WebRTC bağlantısını başlatır.
2. Kullanıcı doğru mikrofonu seçer.
3. Mikrofon sesi backend'e WebRTC audio track olarak gelir.
4. Backend RMS tabanlı konuşma algılama yapar.
5. Kullanıcı sustuğunda ses parçası WAV olarak kaydedilir.
6. WAV dosyası Faster Whisper ile transcribe edilir.
7. Transcript domain normalizer üzerinden düzenlenir.
8. Knowledge base içinde ilgili bilgi aranır.
9. Prompt service, kullanıcı sorusu ve knowledge context ile LLM promptu oluşturur.
10. LLM doğal Türkçe cevap üretir.
11. LLM cevabı kötü, İngilizce veya meta ise KB fallback devreye girer.
12. TTS stream endpoint hazırlanır.
13. Frontend /tts-stream/{response_id} üzerinden sesi streaming olarak oynatır.
14. Metric değerleri SQLite veritabanına kaydedilir.
```

---

## Kullanılan Teknolojiler

- Python
- FastAPI
- aiortc
- PyAV
- Faster Whisper
- OpenRouter API
- Edge TTS
- SQLite
- HTML / CSS / JavaScript
- WebRTC
- HTTP StreamingResponse

---

## Proje Yapısı

```text
voice-ai-agent-webrtc/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
├── .gitignore
│
├── knowledge_base/
│   └── garenta_kiralama_kosullari.docx
│
├── web/
│   └── index.html
│
├── services/
│   ├── knowledge_base_service.py
│   ├── prompt_service.py
│   ├── transcript_normalizer_service.py
│   ├── kb_answer_service.py
│   ├── stt_service.py
│   ├── llm_service.py
│   └── tts_service.py
│
├── utils/
│   ├── audio_utils.py
│   └── metrics.py
│
└── data/
    ├── audio_chunks/
    ├── tts_outputs/
    └── voice_metrics.db
```

---

## Kurulum

Proje klasörüne girilir:

```powershell
cd C:\Users\ekayn\Desktop\voice-ai-agent-webrtc
```

Sanal ortam oluşturulur:

```powershell
python -m venv .venv
```

Sanal ortam aktif edilir:

```powershell
.venv\Scripts\activate
```

Gerekli paketler yüklenir:

```powershell
pip install -r requirements.txt
```

---

## Ortam Değişkenleri

Proje kök dizininde `.env` dosyası oluşturulmalıdır.

Örnek `.env`:

```env
APP_NAME=Voice AI Agent WebRTC

OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=openrouter/free
LLM_TIMEOUT_SECONDS=15

SYSTEM_PROMPT=Sen Türkçe konuşan kısa, doğal ve net cevap veren bir sesli AI asistansın.

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
KB_DIRECT_ANSWER_ENABLED=true
```

`.env` dosyası GitHub'a gönderilmemelidir.

---

## Çalıştırma

```powershell
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

Tarayıcıdan açılır:

```text
http://127.0.0.1:8001
```

---

## Web Arayüzü

Web arayüzünde:

- Mikrofon seçimi yapılabilir.
- WebRTC bağlantısı başlatılır.
- Sistem otomatik dinleme moduna geçer.
- Kullanıcı doğal şekilde konuşur.
- Kullanıcı sustuğunda backend konuşmayı işler.
- Transcript ve asistan cevabı ekranda gösterilir.
- TTS cevabı streaming olarak audio player üzerinden oynatılır.
- STT, LLM ve TTS metric değerleri dashboard'da görüntülenir.

---

## API Endpointleri

### `GET /`

Web arayüzünü döndürür.

### `GET /health`

Backend sağlık durumunu, aktif bağlantıları, STT/TTS config bilgilerini ve knowledge base durumunu döndürür.

### `GET /config`

Audio, STT, TTS, streaming ve knowledge base konfigürasyonlarını döndürür.

### `POST /offer`

Frontend tarafından oluşturulan WebRTC offer bilgisini alır ve backend WebRTC answer bilgisini döndürür.

### `POST /client-state`

Frontend'in anlık durumunu backend'e bildirir.

Örneğin asistan sesi oynarken backend mikrofon inputlarını ignore eder.

### `GET /latest-response`

Son transcript, normalize edilmiş query, asistan cevabı, TTS stream URL'i ve metric bilgilerini döndürür.

### `GET /tts-stream/{response_id}`

Asistan cevabını HTTP streaming audio response olarak döndürür.

### `GET /knowledge/status`

Knowledge base durumunu ve chunk sayılarını döndürür.

### `GET /knowledge/search?q=...`

Knowledge base içinde arama yapar.

### `POST /knowledge/reload`

Knowledge base dosyalarını yeniden yükler.

### `POST /clear-latest-response`

Son response bilgisini temizler.

### `GET /metrics`

SQLite içindeki son voice interaction metric kayıtlarını döndürür.

---

## Knowledge Base

Bu versiyonda knowledge base olarak `knowledge_base/garenta_kiralama_kosullari.docx` dosyası kullanılmıştır.

Servis, DOCX dosyasını okuyup başlıklara göre parçalara böler. Kullanıcı sorusuna göre ilgili chunk'ları bulur ve LLM'e context olarak gönderir.

Örnek desteklenen süreç soruları:

```text
Ek sürücü ekleyebilir miyim?
Aracı geç teslim edersem ne olur?
Eksik yakıtla iade edersem ne olur?
Ödeme için banka kartı kullanabilir miyim?
Kaza yaparsam ne yapmam gerekir?
```

---

## Prompt Service

Prompt service, kullanıcı sorusunu ve knowledge base içeriğini LLM'e uygun hale getirir.

Amaç:

- Cevabı sadece dokümana göre üretmek
- Dokümandaki metni aynen kopyalamamak
- Cevabı doğal müşteri temsilcisi diliyle vermek
- Türkçe, kısa ve sesli okunabilir cevap üretmek
- Bilgi yoksa uydurmamak

---

## Transcript Normalizer

STT çıktısı bazen domain kelimelerinde küçük hatalar yapabilir. Transcript normalizer, Garenta süreçlerine ait bazı ifadeleri normalize ederek knowledge base aramasının daha doğru çalışmasına yardımcı olur.

Örnek domain ifadeleri:

```text
ek sürücü
geç teslim
eksik yakıt
ödeme ve teminat
kaza veya hasar
iptal ve iade
kilometre limiti
```

Bu katman ana cevap üretici değildir. Sadece STT çıktısını arama için daha kullanılabilir hale getirmeye yardımcı olur.

---

## LLM Cevap Güvenliği

LLM bazen meta cevap veya İngilizce analiz döndürebildiği için kalite kontrol eklenmiştir.

Filtrelenen örnekler:

```text
We need to answer...
User asks...
The info includes...
Must answer in Turkish...
Knowledge base says...
```

Bu tarz cevaplar kullanıcıya gösterilmez. Böyle bir durumda KB fallback cevabı devreye girer.

---

## KB Fallback

Asıl hedef, cevabın LLM tarafından knowledge base'e bağlı ve doğal Türkçe şekilde üretilmesidir.

Ancak LLM:

- Boş cevap dönerse
- İngilizce/meta cevap üretirse
- Ham doküman metni gibi cevap verirse
- Bilgi olduğu halde “bulamadım” derse

KB fallback devreye girer ve kullanıcıya kısa, güvenli, dokümana uygun cevap döndürülür.

---

## TTS Streaming

TTS tarafı HTTP streaming olarak çalışır.

```text
LLM cevabı
→ Edge TTS audio chunk üretir
→ FastAPI StreamingResponse
→ Frontend audio player
```

TTS çıktısı tamamlanmış MP3 dosyası beklenmeden frontend'e aktarılır.

---

## STT Durumu

STT tarafı şu an gerçek zamanlı streaming değildir.

Mevcut yapı:

```text
Konuşma algılanır
→ kullanıcı susar
→ ses segmenti WAV olarak kaydedilir
→ Faster Whisper transcribe eder
```

Sonraki aşamada STT tarafı chunk bazlı veya gerçek streaming STT servisi ile geliştirilebilir.

---

## Metric Alanları

Her voice interaction için aşağıdaki değerler tutulur:

- Transcript
- Asistan cevabı
- STT success
- LLM success
- TTS success
- STT latency
- LLM first token latency
- LLM total latency
- TTS first byte latency
- TTS total latency
- Total pipeline latency
- LLM model
- TTS voice
- Error bilgileri

---

## Feedback Loop Koruması

Asistan sesi oynarken frontend backend'e `assistant_playing=true` bilgisini gönderir. Backend bu durumda mikrofon inputlarını işleme almaz.

Bu sayede:

- Asistanın kendi sesini tekrar kullanıcı konuşması sanması
- Ortam sesinden yanlış STT tetiklenmesi
- Whisper hallucination kaynaklı gereksiz cevap üretimi

azaltılmış olur.

---

## Bilinen Sınırlamalar

- STT tarafı gerçek streaming değildir.
- Faster Whisper segment bazlı çalışır.
- TTS streaming HTTP üzerinden yapılır, WebRTC outbound audio track olarak gönderilmez.
- Frontend son cevabı polling ile takip eder.
- Knowledge search şu an keyword / intent skor tabanlıdır.
- Semantic embedding tabanlı RAG henüz eklenmemiştir.
- OpenRouter free model kullanıldığında latency ve cevap kalitesi değişken olabilir.

---

## Sonraki Geliştirme Adımları

- STT tarafını chunk bazlı pseudo-streaming hale getirmek
- Partial transcript desteği eklemek
- Semantic embedding tabanlı RAG eklemek
- Qdrant veya benzeri vector database entegrasyonu
- LLM token stream çıktısını TTS'e daha erken aktarmak
- Cümle bazlı TTS streaming pipeline kurmak
- HTTP streaming yerine WebRTC outbound audio track değerlendirmek
- VAD tabanlı daha sağlam konuşma algılama eklemek
- Frontend polling yerine WebSocket veya DataChannel kullanmak
- Daha detaylı latency breakdown dashboard'u eklemek

---

## Güncel Durum

Bu versiyonda sistem çalışır durumdadır.

Tamamlananlar:

- WebRTC microphone input
- Mikrofon seçimi
- Otomatik konuşma algılama
- Faster Whisper STT
- Whisper preload
- Knowledge base service
- Prompt service
- Transcript normalizer
- OpenRouter LLM
- LLM kalite kontrol
- KB fallback
- Edge TTS
- Streaming TTS endpoint
- Frontend audio playback
- Feedback loop guard
- SQLite metrics
- Profesyonel dashboard UI

Devam eden / sonraki konu:

- STT tarafının gerçek streaming veya chunk bazlı partial transcript yapısına dönüştürülmesi
- Knowledge base search yapısının semantic RAG mimarisine taşınması