Это критически важный компонент, который связывает современный WebRTC-стек (LiveKit) с классической телефонией (SIP/Asterisk). Ошибки здесь приводят к задержкам (jitter), потере пакетов и невозможности корректной работы VAD/ASR. Мы реализуем это с использованием производственных (production-grade) практик.

---

# 🚀 ЭТАП 0.2: Инфраструктура телефонии и WebRTC-мост

## Шаг 1: Конфигурация Asterisk (SIP-ядро, AMD, SIPREC)

Asterisk будет выступать в роли SIP-сервера (PBX), принимающего звонки от нашего Rust-моста и маршрутизирующего их во внешнюю сеть (SIP-транк), одновременно запуская детекцию автоответчика (AMD) и запись разговора.

Создаем директорию `infra/asterisk/config/` и добавляем файлы.

### 1.1. Настройка PJSIP (`pjsip.conf`)
Определяем эндпоинт для нашего Rust-моста и транк к провайдеру.

```ini
; infra/asterisk/config/pjsip.conf

; --- Транспорт для WebRTC/SIP моста (внутренняя сеть) ---
[transport-webrtc-bridge]
type=transport
protocol=udp
bind=0.0.0.0:5060

; --- Endpoint для Rust WebRTC-SIP Bridge ---
[webrtc-bridge]
type=endpoint
context=from-bridge
disallow=all
allow=opus,ulaw,alaw ; Opus предпочтителен для AI (широкая полоса), ulaw/alaw как fallback
dtmf_mode=rfc4733
direct_media=no
webrtc=yes ; Включаем поддержку WebRTC-фич, если мост использует их
rtp_symmetric=yes
force_rport=yes

[webrtc-bridge-auth]
type=auth
auth_type=userpass
username=voicegraph_bridge
password=SuperSecretBridgePassword2026!

[webrtc-bridge-aor]
type=aor
max_contacts=100
remove_existing=yes

; --- SIP Транк к внешнему провайдеру (Пример) ---
[outbound-trunk]
type=endpoint
context=from-trunk
disallow=all
allow=opus,ulaw,alaw
outbound_auth=trunk-auth
aors=trunk-aor

[trunk-auth]
type=auth
auth_type=userpass
username=YOUR_PROVIDER_LOGIN
password=YOUR_PROVIDER_PASSWORD

[trunk-aor]
type=aor
contact=sip:provider.example.com:5060
```

### 1.2. Диалплан с AMD и Записью (`extensions.conf`)
Здесь мы реализуем логику: определение номера, запуск AMD, и если это человек — передача управления в контекст для AI (или запись через SIPREC/MixMonitor).

```ini
; infra/asterisk/config/extensions.conf

[from-bridge]
; Маршрутизация исходящих звонков от AI-моста
exten => _X.,1,NoOp(=== VoiceGraph AI Call Initiated: ${EXTEN} ===)
; 1. Инициализация записи разговора (Dual-channel для последующего разделения ролей ASR)
; Сохраняем в формате wav, который позже скрипт загрузит в MinIO
 same => n,Set(RECORDING_FILE=/var/spool/asterisk/monitor/${STRFTIME(${EPOCH},,%Y%m%d-%H%M%S)}-${EXTEN}-${UNIQUEID})
 same => n,MixMonitor(${RECORDING_FILE}.wav,b) ; 'b' = both channels mixed, или используйте 'W(,)w(,)' для раздельных каналов

; 2. Запуск Answering Machine Detection (AMD)
; Параметры: InitialSilence, Greeting, AfterGreetingSilence, TotalAnalysisTime, MinimumWordLength, BetweenWordsSilence, MaximumNumberOfWords, SilenceThreshold, MaximumWordLength
 same => n,AMD(2500,1500,1000,5000,100,100,3,256,5000)

; 3. Анализ результата AMD
 same => n,GotoIf($["${AMDSTATUS}" = "MACHINE"]?machine_detected,1)
 same => n,GotoIf($["${AMDSTATUS}" = "HUMAN"]?human_detected,1)
 same => n,GotoIf($["${AMDSTATUS}" = "NOTSURE"]?human_detected,1) ; Лучше ошибиться в пользу человека

; Ветка: Обнаружен автоответчик
exten => machine_detected,1,NoOp(AMD Detected MACHINE. Hanging up to save AI costs.)
 same => n,Hangup()

; Ветка: Обнаружен человек (или не определено) -> Передача в AI
exten => human_detected,1,NoOp(AMD Detected HUMAN or NOTSURE. Proceeding to AI handler.)
; Здесь мы можем передать звонок в очередь LiveKit/AI или просто держать линию, 
; так как медиа-поток уже идет через мост в LiveKit.
 same => n,Wait(86400) ; Держим линию, пока WebRTC-мост не положит трубку
 same => n,Hangup()

; Обработка завершения звонка для триггера загрузки в MinIO
exten => h,1,NoOp(Call Ended. Triggering MinIO upload script for ${RECORDING_FILE}.wav)
 same => n,System(/usr/local/bin/upload_to_minio.sh ${RECORDING_FILE}.wav ${EXTEN} ${UNIQUEID} &)
```

---

## Шаг 2: Rust WebRTC-to-SIP Bridge (Архитектура и ядро)

Полноценный SIP-стек на Rust — это сложная задача. Мы будем использовать комбинацию проверенных крейтов: `rsip` (для SIP-сигналинга) и `webrtc` (для медиа-потоков), работающих асинхронно через `tokio`.

*Примечание: Это production-ready скелет, демонстрирующий правильную архитектуру медиа-релея и обработки сигналов.*

**Файл: `bridge/Cargo.toml`**
```toml
[package]
name = "voicegraph-webrtc-sip-bridge"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1.38", features = ["full"] }
webrtc = "0.9.0"          # WebRTC стек
rsip = "0.4.0"            # SIP протокол
rsip-tokio = "0.4.0"      # Асинхронный транспорт для rsip
tracing = "0.1"
tracing-subscriber = "0.3"
anyhow = "1.0"
uuid = { version = "1.8", features = ["v4"] }
```

**Файл: `bridge/src/main.rs` (Ключевая логика моста)**
```rust
use anyhow::Result;
use rsip::{prelude::*, Request, Response, Method};
use rsip_tokio::{UdpTransport, UdpTransportConfig};
use std::net::SocketAddr;
use tracing::{info, warn, error};
use webrtc::api::APIBuilder;
use webrtc::peer_connection::configuration::RTCConfiguration;

// 1. Конфигурация моста
struct BridgeConfig {
    asterisk_addr: SocketAddr,
    bridge_username: String,
    bridge_password: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("🚀 Запуск VoiceGraph WebRTC-to-SIP Bridge...");

    let config = BridgeConfig {
        asterisk_addr: "127.0.0.1:5060".parse()?,
        bridge_username: "voicegraph_bridge".into(),
        bridge_password: "SuperSecretBridgePassword2026!".into(),
    };

    // 2. Инициализация SIP транспорта
    let mut transport = UdpTransport::listen("0.0.0.0:5061".parse()?, UdpTransportConfig::default()).await?;
    info!("📞 SIP транспорт запущен на 0.0.0.0:5061");

    // 3. Основной цикл обработки входящих WebRTC сессий (упрощенно: триггер от LiveKit)
    // В реальности здесь будет gRPC/WebSocket слушатель от LiveKit Agents
    loop {
        // Эмуляция получения команды на исходящий звонок от LiveKit Orchestrator
        // let call_request = listen_livekit_trigger().await?;
        
        // Для примера: инициируем звонок на тестовый номер
        let target_number = "79001234567";
        info!("📲 Инициация звонка на: {}", target_number);
        
        if let Err(e) = handle_outbound_call(&mut transport, &config, target_number).await {
            error!("Ошибка обработки звонка: {}", e);
        }
    }
}

async fn handle_outbound_call(
    transport: &mut UdpTransport,
    config: &BridgeConfig,
    target: &str,
) -> Result<()> {
    // 1. Формирование SIP INVITE
    let invite = build_invite_request(config, target)?;
    
    // 2. Отправка INVITE в Asterisk
    transport.send(&invite.into()).await?;
    info!("📤 Отправлен SIP INVITE в Asterisk");

    // 3. Ожидание ответа (100 Trying, 180 Ringing, 200 OK)
    // В продакшене здесь нужен State Machine (FSM) для обработки таймаутов и retransmissions
    while let Some(msg) = transport.recv().await? {
        if let rsip::Message::Response(res) = msg {
            match res.status_code() {
                rsip::StatusCode::Trying => info!("⏳ 100 Trying"),
                rsip::StatusCode::Ringing => info!("🔔 180 Ringing"),
                rsip::StatusCode::Ok => {
                    info!("✅ 200 OK. Соединение установлено.");
                    // 4. Отправка SIP ACK
                    let ack = build_ack_request(&invite, &res)?;
                    transport.send(&ack.into()).await?;
                    
                    // 5. ЗАПУСК МЕДИA-РЕЛЕЯ (WebRTC <-> RTP)
                    // Здесь инициализируется PeerConnection из crate `webrtc`
                    // и начинается пересылка RTP-пакетов между Asterisk и LiveKit
                    start_media_relay(res).await?;
                    
                    break;
                }
                _ => {
                    warn!("⚠️ Получен неожиданный ответ: {}", res.status_code());
                    break;
                }
            }
        }
    }
    Ok(())
}

fn build_invite_request(config: &BridgeConfig, target: &str) -> Result<Request> {
    // Упрощенная сборка INVITE. В продакшене использовать rsip::services::RequestBuilder
    // с корректными Via, From, To, Call-ID, CSeq, SDP (с Opus/PCMU кодеками)
    let sdp = generate_sdp_offer(); // Функция генерации SDP с поддержкой Opus и DTMF (RFC4733)
    
    let mut req = Request::builder()
        .method(Method::Invite)
        .uri(format!("sip:{}@{}", target, config.asterisk_addr))
        .header(("From", format!("<sip:{}@{}>;tag=vg123", config.bridge_username, config.asterisk_addr.ip())))
        .header(("To", format!("<sip:{}@{}>", target, config.asterisk_addr.ip())))
        .header(("Call-ID", uuid::Uuid::new_v4().to_string()))
        .header(("CSeq", "1 INVITE"))
        .header(("Content-Type", "application/sdp"))
        .body(sdp)
        .build()?;
        
    // Добавление Digest Authentication (упрощенно)
    // req.authorize(&config.bridge_username, &config.bridge_password);
    
    Ok(req)
}

async fn start_media_relay(ok_response: rsip::Response) -> Result<()> {
    info!("🌐 Запуск WebRTC <-> RTP медиа-релея...");
    // 1. Парсинг SDP из 200 OK для получения IP/Port и выбранных кодеков Asterisk
    // 2. Инициализация webrtc::peer_connection::RTCPeerConnection
    // 3. Настройка on_track callback для перенаправления RTP-пакетов в UDP-сокет Asterisk
    // 4. Настройка VAD-триггеров для прерывания (Barge-in)
    Ok(())
}

fn generate_sdp_offer() -> String {
    // Минимальный SDP для Asterisk, требующий Opus и ulaw, с поддержкой DTMF
    r#"v=0
o=- 123456 1 IN IP4 127.0.0.1
s=VoiceGraph Bridge
c=IN IP4 127.0.0.1
t=0 0
m=audio 9 RTP/AVP 111 0
a=rtpmap:111 opus/48000/2
a=rtpmap:0 PCMU/8000
a=rtcp-fb:111 transport-cc
a=sendrecv
"#.to_string()
}
```

---

## Шаг 3: Инфраструктурная оркестрация (Docker Compose)

Собираем всё вместе для локальной разработки и тестирования.

**Файл: `docker-compose.telephony.yml`**
```yaml
version: '3.8'

services:
  # 1. Asterisk PBX
  asterisk:
    image: asterisk/asterisk:20-certified
    container_name: voicegraph-asterisk
    volumes:
      - ./infra/asterisk/config:/etc/asterisk:ro
      - ./infra/asterisk/recordings:/var/spool/asterisk/monitor
      - ./infra/asterisk/logs:/var/log/asterisk
    ports:
      - "5060:5060/udp"   # SIP
      - "10000-10100:10000-10100/udp" # RTP диапазон
    networks:
      - voicegraph-net
    command: ["-U", "asterisk", "-G", "asterisk", "-f"]

  # 2. Rust WebRTC-to-SIP Bridge
  webrtc-bridge:
    build:
      context: ./bridge
      dockerfile: Dockerfile
    container_name: voicegraph-bridge
    environment:
      - RUST_LOG=info
      - ASTERISK_ADDR=asterisk:5060
    ports:
      - "5061:5061/udp"
    depends_on:
      - asterisk
    networks:
      - voicegraph-net

  # 3. MinIO (для хранения записей разговоров)
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin123
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio-data:/data
    networks:
      - voicegraph-net

networks:
  voicegraph-net:
    driver: bridge

volumes:
  minio-data:
```

*Скрипт-хук для Asterisk (`infra/asterisk/upload_to_minio.sh`)*:
```bash
#!/bin/bash
# Принимает: FILE_PATH EXTENSION UNIQUE_ID
FILE=$1
EXT=$2
UID=$3
# Ожидание завершения записи файла Asterisk
sleep 2
mc alias set myminio http://minio:9000 minioadmin minioadmin123
mc cp $FILE myminio/voicegraph-recordings/calls/$UID.wav
rm -f $FILE # Очистка локального диска
```

---

## Шаг 4: Нагрузочное тестирование шлюза (100 параллельных каналов)

Чтобы гарантировать, что мост не "упадет" и не внесет задержки, превышающие 800 мс, мы используем **SIPp** для эмуляции нагрузки.

### 4.1. Сценарий SIPp (`uac_100_calls.xml`)
```xml
<?xml version="1.0" encoding="ISO-8859-1" ?>
<!DOCTYPE scenario SYSTEM "sipp.dtd">
<scenario name="VoiceGraph Load Test UAC">
  <send retrans="500">
    <![CDATA[
      INVITE sip:79001234567@[remote_ip]:[remote_port] SIP/2.0
      Via: SIP/2.0/[transport] [local_ip]:[local_port];branch=[branch]
      From: "TestLoad" <sip:loadtest@[local_ip]:[local_port]>;tag=[call_number]
      To: <sip:79001234567@[remote_ip]:[remote_port]>
      Call-ID: [call_id]
      CSeq: 1 INVITE
      Contact: sip:[local_ip]:[local_port]
      Max-Forwards: 70
      Subject: Load Test
      Content-Type: application/sdp
      Content-Length: [len]

      v=0
      o=user1 53655765 2353687637 IN IP[local_ip_type] [local_ip]
      s=-
      c=IN IP[media_ip_type] [media_ip]
      t=0 0
      m=audio [media_port] RTP/AVP 111 0
      a=rtpmap:111 opus/48000/2
      a=rtpmap:0 PCMU/8000
      a=sendrecv
    ]]>
  </send>

  <recv response="100" optional="true"></recv>
  <recv response="180" optional="true"></recv>
  <recv response="200" rtd="true" crlf="true"></recv>

  <send>
    <![CDATA[
      ACK sip:79001234567@[remote_ip]:[remote_port] SIP/2.0
      Via: SIP/2.0/[transport] [local_ip]:[local_port];branch=[branch]
      From: "TestLoad" <sip:loadtest@[local_ip]:[local_port]>;tag=[call_number]
      To: <sip:79001234567@[remote_ip]:[remote_port]>[peer_tag_param]
      Call-ID: [call_id]
      CSeq: 1 ACK
      Contact: sip:[local_ip]:[local_port]
      Max-Forwards: 70
      Content-Length: 0
    ]]>
  </send>

  <pause milliseconds="5000"/> <!-- Имитация 5-секундного разговора -->

  <send retrans="500">
    <![CDATA[
      BYE sip:79001234567@[remote_ip]:[remote_port] SIP/2.0
      Via: SIP/2.0/[transport] [local_ip]:[local_port];branch=[branch]
      From: "TestLoad" <sip:loadtest@[local_ip]:[local_port]>;tag=[call_number]
      To: <sip:79001234567@[remote_ip]:[remote_port]>[peer_tag_param]
      Call-ID: [call_id]
      CSeq: 2 BYE
      Contact: sip:[local_ip]:[local_port]
      Max-Forwards: 70
      Content-Length: 0
    ]]>
  </send>
  <recv response="200" crlf="true"></recv>
</scenario>
```

### 4.2. Запуск теста и мониторинг
1. Запускаем `docker-compose -f docker-compose.telephony.yml up -d`.
2. Запускаем SIPp (извне или из отдельного контейнера):
   ```bash
   sipp -sf uac_100_calls.xml -s 79001234567 -r 10 -rp 1000 -l 100 -m 100 127.0.0.1:5060
   ```
   * `-r 10`: 10 звонков в секунду.
   * `-l 100`: Максимум 100 одновременных каналов.
   * `-m 100`: Всего 100 звонков.
3. **Мониторинг**: В отдельном окне используем `docker stats` и `asterisk -rx "core show channels"` для отслеживания:
   - Потребление CPU мостом (не должно превышать 30-40% на ядро при 100 каналах).
   - Отсутствие "dropped packets" в выводе `sipp` в конце теста.

---

## ✅ Definition of Done (Критерии готовности Подзадачи 0.2)

Прежде чем перейти к **Подзадаче 0.3 (Pre-processing и маскирование PII)**, убедитесь, что:

- [ ] Asterisk успешно принимает SIP INVITE от Rust-моста и возвращает `200 OK`.
- [ ] Функция `AMD()` в `extensions.conf` корректно отрабатывает (можно проверить по логам Asterisk: `AMD: HUMAN` или `AMD: MACHINE`).
- [ ] Файлы записи разговоров (`.wav`) успешно создаются в `/var/spool/asterisk/monitor` и скрипт-хук корректно загружает их в MinIO.
- [ ] Нагрузочный тест `sipp` на 100 параллельных каналов завершен с 0% потерь (Lost packets = 0) и без падения контейнеров.
- [ ] Rust-код скомпилирован без предупреждений (`cargo build --release`), линтер (`cargo clippy`) не выдает ошибок.
