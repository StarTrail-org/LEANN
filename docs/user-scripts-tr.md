# Kullanıcı Scriptleri: Günlük Yaşamda LEANN

Bu dokümantasyon, LEANN'ı günlük yaşamda kullanmak için hazırlanmış otomasyon scriptlerini açıklar.

## Kurulum

### 1. Scriptleri İndirme

Bu scriptleri kullanmak için önce LEANN repository'sini klonlayın:

```bash
git clone https://github.com/yichuan-w/LEANN.git
cd LEANN
```

### 2. Scriptleri ~/bin Klasörüne Kopyalama

```bash
# ~/.bin klasörü oluşturma (yoksa)
mkdir -p ~/bin

# Scriptleri kopyalama
cp bin/leann-sync-all.sh ~/bin/
cp bin/leann-sync-dev.sh ~/bin/
cp bin/leann-sync-personal.sh ~/bin/
cp bin/leann-sync-brave.sh ~/bin/
cp bin/leann-sync-mail.sh ~/bin/
cp bin/leann-sync-imessage.sh ~/bin/
cp bin/leann-sync-calendar.sh ~/bin/

# Çalıştırılabilir yapma
chmod +x ~/bin/leann-*.sh
```

### 3. PATH'e Ekleme

```bash
# ~/.zshrc veya ~/.bashrc dosyanıza ekleyin
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### 4. Ollama Kurulumu (Embedding için)

```bash
# Ollama'yı başlatma
ollama serve &

# Embedding modeli indirme
ollama pull nomic-embed-text
```

### 5. LEANN Kurulumu

```bash
cd LEANN
uv sync --extra diskann
```

## Kullanım

### Hızlı Başlangıç

```bash
# Tüm indexleri güncelle
leann-sync-all.sh
```

### Bireysel Scriptler

| Script | Açıklama |
|--------|-----------|
| `leann-sync-all.sh` | Tüm indexleri sırayla günceller |
| `leann-sync-dev.sh` | Geliştirme ortamı kodlarını indeksler |
| `leann-sync-personal.sh` | Kişisel belgeleri (Documents, Nextcloud) indeksler |
| `leann-sync-brave.sh` | Brave tarayıcı geçmişini indeksler |
| `leann-sync-mail.sh` | Apple Mail e-postalarını indeksler |
| `leann-sync-imessage.sh` | iMessage mesajlarını indeksler |
| `leann-sync-calendar.sh` | Apple Calendar etkinliklerini indeksler |

### Örnek Kullanım Senaryoları

#### Senaryo 1: Günlük Geliştirme İş Akışı

```bash
# Her sabah geliştirme ortamınızı güncelleyin
leann-sync-dev.sh
```

Bu komut:
- ~/Development klasöründeki kodlarınızı tarar
- AST-aware chunking ile kod yapısını korur
- DiskANN backend ile indeks oluşturur

#### Senaryo 2: Kişisel Doküman Arama

```bash
# Kişisel belgelerinizi indeksleyin
leann-sync-personal.sh
```

Bu komut:
- ~/Documents, ~/Nextcloud, ~/Nextcloud2 klasörlerini tarar
- Tüm belgeleri (PDF, TXT, MD, Word, Excel, PowerPoint) indeksler

#### Senaryo 3: Tarayıcı Geçmişi Arama

```bash
# Brave tarayıcı geçmişinizi indeksleyin
leann-sync-brave.sh
```

#### Senaryo 4: E-posta Arama

```bash
# Apple Mail e-postalarınızı indeksleyin
leann-sync-mail.sh
```

#### Senaryo 5: iMessage Arama

```bash
# iMessage mesajlarınızı indeksleyin
leann-sync-imessage.sh
```

## Script Özelleştirme

### Kendi Scriptinizi Oluşturma

```bash
#!/bin/bash
# my-custom-index.sh

export LEANN_HOME="$HOME/.leann"
export OLLAMA_HOST="http://localhost:11434"

leann build my-custom-index \
  --docs ~/MyDocuments \
  --embedding-mode ollama \
  --embedding-model nomic-embed-text \
  --backend-name diskann \
  --force
```

### Parametre Değiştirme

Scriptlerdeki parametreleri kendi ihtiyaçlarınıza göre değiştirebilirsiniz:

```bash
# Daha büyük chunk boyutu
--doc-chunk-size 2048

# Embedding modeli değiştirme
--embedding-model BAAI/bge-base-en-v1.5

# HNSW backend kullanma
--backend-name hnsw
```

## Sıkça Sorulan Sorular

### S: "leann: command not found" hatası alıyorum

C: LEANN kurulumunun PATH'e eklendiğinden emin olun:
```bash
export PATH="/path/to/LEANN/packages/leann-core:$PATH"
```

### S: Ollama bağlantısı başarısız

C: Ollama'nın çalıştığını kontrol edin:
```bash
ollama serve &
ollama list
```

### S: Index oluşturma çok yavaş

C: Daha küçük bir dataset ile test edin:
```bash
--max-count 1000
```

## İleri Düzey Kullanım

### LEANN CLI Komutları

```bash
# Index oluşturma
leann build my-index --docs ./documents

# Arama
leann search my-index "arama sorgusu"

# Soru sorma
leann ask my-index --interactive

# Indexleri listeleme
leann list

# Index kaldırma
leann remove my-index
```

### Embedding Modları

| Mod | Açıklama |
|-----|-----------|
| `sentence-transformers` | HuggingFace modelleri |
| `openai` | OpenAI API |
| `mlx` | Apple Silicon MLX |
| `ollama` | Yerel Ollama |

### Backend Seçimi

| Backend | Kullanım |
|---------|----------|
| `hnsw` | Küçük-orta ölçekli (<10M vektör) |
| `diskann` | Büyük ölçekli, disk tabanlı |

---

Bu dokümantasyon, LEANN'ın günlük kullanımını kolaylaştırmak için hazırlanmıştır. Herhangi bir sorunuz varsa, lütfen GitHub Issues üzerinden sorun.
