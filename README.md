# Voice AI Agent WebRTC

Bu proje, WebRTC tabanlı çalışan basit bir sesli AI agent uygulamasıdır.

Kullanıcı tarayıcı üzerinden mikrofona konuşur. Ses backend tarafına WebRTC ile gönderilir. Backend tarafında konuşma algılanır, ses yazıya çevrilir, LLM ile cevap üretilir ve bu cevap TTS ile tekrar sese çevrilir. Sonuç web arayüzünde yazı ve ses olarak gösterilir.

## Projenin Amacı

Bu projede amaç, Telegram bot dışında çalışan ve sadece sesli mod üzerinden ilerleyen bir AI agent yapısı oluşturmaktır.

Temel akış şu şekildedir:

```text
Mikrofon → WebRTC → STT → LLM → TTS → Web arayüzünde sesli cevap
```

## Kullanılan Teknolojiler

Projede kullanılan başlıca teknolojiler:

```text
Python
FastAPI
WebRTC
aiortc
Faster Whisper
OpenRouter
Edge TTS
SQLite
HTML / CSS / JavaScript
```

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
├── web/
│   └── index.html
│
├── services/
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

## Temel Özellikler

- Tarayıcıdan mikrofon izni alınır.
- Ses WebRTC ile backend'e gönderilir.
- Kullanıcının konuşmaya başladığı ve sustuğu algılanır.
- Ses dosyası WAV olarak kaydedilir.
- STT ile konuşma yazıya çevrilir.
- LLM ile kullanıcıya cevap oluşturulur.
- TTS ile cevap ses dosyasına çevrilir.
- Web arayüzünde cevap yazı ve ses olarak gösterilir.
- STT, LLM ve TTS süreleri metric olarak kaydedilir.
- Metric kayıtları SQLite veritabanında tutulur.

## Kurulum

Projeyi bilgisayara aldıktan sonra klasöre girilir:

```bash
cd voice-ai-agent-webrtc
```

Virtual environment oluşturulur:

```bash
python -m venv .venv
```

Windows PowerShell için aktif edilir:

```bash
.venv\Scripts\activate
```

Gerekli paketler yüklenir:

```bash
pip install -r requirements.txt
```

## .env Ayarları

Projenin çalışması için ana klasörde `.env` dosyası olmalıdır.

Örnek `.env` içeriği:

```env
APP_NAME=Voice AI Agent WebRTC

OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_MODEL=openrouter/free
LLM_TIMEOUT_SECONDS=60

SYSTEM_PROMPT=Sen sesli çalışan yardımcı bir AI agentsın. Kullanıcıya Türkçe, kısa, doğal ve net cevap ver.

STT_PROVIDER=faster_whisper
WHISPER_MODEL_SIZE=small
WHISPER_LANGUAGE=tr

TARGET_SAMPLE_RATE=16000
AUDIO_OUTPUT_DIR=data/audio_chunks
MIN_AUDIO_RMS=50

SPEECH_RMS_THRESHOLD=120
SILENCE_END_SECONDS=1.2
MIN_UTTERANCE_SECONDS=1.0
MAX_UTTERANCE_SECONDS=12
PRE_SPEECH_SECONDS=0.3

TTS_PROVIDER=edge_tts
TTS_VOICE=tr-TR-EmelNeural
TTS_RATE=+0%
TTS_VOLUME=+0%
TTS_OUTPUT_DIR=data/tts_outputs

METRICS_DB_PATH=data/voice_metrics.db
```

`.env` dosyası GitHub'a yüklenmemelidir. Çünkü içinde API key gibi gizli bilgiler bulunabilir.

## Projeyi Çalıştırma

Aşağıdaki komut ile FastAPI server başlatılır:

```bash
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

Sonrasında tarayıcıdan şu adres açılır:

```text
http://127.0.0.1:8001
```

Arayüzde `WebRTC Başlat` butonuna basılır ve mikrofona konuşulur.

## Çalışma Akışı

Uygulama şu şekilde çalışır:

```text
1. Kullanıcı web arayüzünden WebRTC bağlantısını başlatır.
2. Mikrofon sesi backend'e gönderilir.
3. Backend konuşmayı algılar.
4. Kullanıcı sustuğunda ses dosyası kaydedilir.
5. Ses STT ile text'e çevrilir.
6. Text LLM'e gönderilir.
7. LLM cevabı TTS ile sese çevrilir.
8. Cevap web arayüzünde yazı ve ses olarak gösterilir.
9. Süre bilgileri SQLite'a kaydedilir.
```

## Metric Kayıtları

Projede her konuşma için bazı metric bilgileri tutulur.

Kaydedilen bilgiler:

```text
STT latency
LLM first token time
LLM total time
TTS first byte time
TTS total time
Total pipeline time
Transcript
Answer
LLM model
TTS voice
```

Bu kayıtlar `data/voice_metrics.db` içinde tutulur.

Web arayüzünde de son metric kayıtları tablo olarak gösterilir.

## API Endpointleri

Projede kullanılan bazı endpointler:

```text
GET /
Web arayüzünü açar.

GET /health
Sistemin çalışıp çalışmadığını gösterir.

GET /config
Proje ayarlarını gösterir.

POST /offer
WebRTC offer bilgisini backend'e gönderir.

GET /latest-response
Son kullanıcı mesajını, AI cevabını ve ses dosyasını döndürür.

GET /metrics
Son metric kayıtlarını döndürür.
```

## Örnek Kullanım

Kullanıcı şunu söyleyebilir:

```text
Ders çalışmam lazım, bana bir plan yap.
```

Asistan örnek olarak şöyle cevap verebilir:

```text
Ders çalışman için günlük 2 saatlik bloklar ayır. Her blokta bir konuyu 25 dakika çalış ve 5 dakika dinlen. Öğrendiklerini kısa bir özetle not al ve günün sonunda 10 dakika tekrar et.
```

## GitHub'a Yüklenmemesi Gerekenler

Aşağıdaki dosyalar GitHub'a yüklenmemelidir:

```text
.env
.venv/
data/
*.db
*.wav
*.mp3
__pycache__/
*.pyc
```

Bunun için `.gitignore` dosyası kullanılmaktadır.

## Projenin Şu Anki Durumu

Şu an projede temel MVP akışı çalışmaktadır.

Çalışan kısımlar:

```text
WebRTC ses gönderimi
Konuşma algılama
STT
LLM
TTS
Web arayüzünde cevap gösterme
Sesli cevap oynatma
Metric kaydı
```

## Geliştirilebilecek Kısımlar

İleride geliştirilebilecek bazı noktalar:

```text
Cevabı direkt WebRTC üzerinden ses olarak döndürmek
Daha gelişmiş VAD kullanmak
WebSocket ile anlık durum güncellemek
Daha iyi model fallback sistemi kurmak
Session memory eklemek
Admin panel yapmak
Deployment ayarlarını eklemek
```

## Not

Bu proje staj sürecinde WebRTC, STT, LLM, TTS ve metric yapısını öğrenmek ve temel bir voice AI agent akışı oluşturmak için geliştirilmiştir.