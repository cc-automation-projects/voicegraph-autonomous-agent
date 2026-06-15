use anyhow::Result;
use rsip::{prelude::*, Request, Method};
use rsip_tokio::{UdpTransport, UdpTransportConfig};
use std::net::SocketAddr;
use tracing::{info, warn, error};
use uuid::Uuid;

struct BridgeConfig {
    asterisk_addr: SocketAddr,
    bridge_username: String,
    bridge_password: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("Запуск VoiceGraph WebRTC-to-SIP Bridge...");

    let config = BridgeConfig {
        asterisk_addr: "127.0.0.1:5060".parse()?,
        bridge_username: "voicegraph_bridge".into(),
        bridge_password: "SuperSecretBridgePassword2026!".into(),
    };

    let mut transport = UdpTransport::listen("0.0.0.0:5061".parse()?, UdpTransportConfig::default()).await?;
    info!("SIP транспорт запущен на 0.0.0.0:5061");

    loop {
        let target_number = "79001234567";
        info!("Инициация звонка на: {}", target_number);

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
    let invite = build_invite_request(config, target)?;
    transport.send(&invite.into()).await?;
    info!("Отправлен SIP INVITE в Asterisk");

    while let Some(msg) = transport.recv().await? {
        if let rsip::Message::Response(res) = msg {
            match res.status_code() {
                rsip::StatusCode::Trying => info!("100 Trying"),
                rsip::StatusCode::Ringing => info!("180 Ringing"),
                rsip::StatusCode::Ok => {
                    info!("200 OK. Соединение установлено.");
                    let ack = build_ack_request(&invite, &res)?;
                    transport.send(&ack.into()).await?;
                    start_media_relay(res).await?;
                    break;
                }
                _ => {
                    warn!("Получен неожиданный ответ: {}", res.status_code());
                    break;
                }
            }
        }
    }
    Ok(())
}

fn build_invite_request(config: &BridgeConfig, target: &str) -> Result<Request> {
    let sdp = generate_sdp_offer();

    let mut req = Request::builder()
        .method(Method::Invite)
        .uri(format!("sip:{}@{}", target, config.asterisk_addr))
        .header(("From", format!("<sip:{}@{}>;tag=vg{}", config.bridge_username, config.asterisk_addr.ip(), Uuid::new_v4().to_string().split('-').next().unwrap_or("x"))))
        .header(("To", format!("<sip:{}@{}>", target, config.asterisk_addr.ip())))
        .header(("Call-ID", Uuid::new_v4().to_string()))
        .header(("CSeq", "1 INVITE"))
        .header(("Content-Type", "application/sdp"))
        .body(sdp)
        .build()?;

    Ok(req)
}

fn build_ack_request(invite: &Request, response: &rsip::Response) -> Result<Request> {
    let mut ack = Request::builder()
        .method(Method::Ack)
        .uri(invite.uri().clone())
        .header(("From", invite.header("From").unwrap().clone()))
        .header(("To", response.header("To").unwrap().clone()))
        .header(("Call-ID", invite.header("Call-ID").unwrap().clone()))
        .header(("CSeq", "1 ACK"))
        .body("")
        .build()?;
    Ok(ack)
}

async fn start_media_relay(_ok_response: rsip::Response) -> Result<()> {
    info!("Запуск WebRTC <-> RTP медиа-релея...");
    Ok(())
}

fn generate_sdp_offer() -> String {
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
"#
    .to_string()
}
