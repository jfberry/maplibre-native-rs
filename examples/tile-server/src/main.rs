//! Tile rendering server with two parallel-Rust caches:
//!
//! 1. **Rendered-PNG cache** keyed by `(z, x, y)` — `try_get_with`
//!    dedupes concurrent requests for the same tile.
//! 2. **Resource cache** keyed by URL — every resource mbgl asks for
//!    (style.json, source TileJSON, vector tiles, glyph PBFs, sprites)
//!    is served by an async `FileSource` callback running on tokio.
//!    The callback hits moka first; on miss it fetches via `reqwest`
//!    or `tokio::fs` and writes the bytes back.
//!
//! Every byte of network/disk I/O happens off the single-threaded C++
//! render thread.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::{Path, State},
    http::{header, StatusCode},
    response::{Html, Response},
    routing::get,
    Router,
};
use bytes::Bytes;
use maplibre_native::{
    register_file_source_callback, FileSourceRequestCallback, FsErrorReason, FsResponse,
    SingleThreadedRenderPool,
};
use moka::future::Cache;

type TileKey = (u8, u32, u32);

/// `Some(bytes)` for a successful body, `None` for a `404`/missing-file
/// miss that mbgl would otherwise re-issue. Errors are not cached so
/// transient failures get retried.
type CachedResource = Option<Bytes>;
type ResourceCache = Cache<String, CachedResource>;

#[derive(Clone)]
struct AppState {
    rendered: Cache<TileKey, Bytes>,
    style: Arc<PathBuf>,
}

async fn render_tile_png(
    style: Arc<PathBuf>,
    (z, x, y): TileKey,
) -> Result<Bytes, RenderError> {
    let image = SingleThreadedRenderPool::global_pool()
        .render_tile((*style).clone(), z, x, y)
        .await?;
    let mut png = Vec::new();
    image
        .as_image()
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)?;
    Ok(Bytes::from(png))
}

async fn rendered_style_tile(
    State(state): State<AppState>,
    Path((z, x, y)): Path<(u8, u32, u32)>,
) -> Result<Response, StatusCode> {
    let key = (z, x, y);
    let bytes = state
        .rendered
        .try_get_with(key, render_tile_png(state.style.clone(), key))
        .await
        .map_err(|e| {
            eprintln!("render failed for {key:?}: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        })?;

    Ok(Response::builder()
        .header(header::CONTENT_TYPE, "image/png")
        .header(header::CACHE_CONTROL, "max-age=3600")
        .body(axum::body::Body::from(bytes))
        .unwrap())
}

async fn index() -> Html<&'static str> {
    Html(include_str!("index.html"))
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests")
        .join("fixtures")
        .join(name)
}

async fn fetch_resource(
    http: &reqwest::Client,
    url: &str,
) -> Result<CachedResource, FetchError> {
    if let Some(path) = url.strip_prefix("file://") {
        match tokio::fs::read(path).await {
            Ok(v) => Ok(Some(Bytes::from(v))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    } else if url.starts_with("http://") || url.starts_with("https://") {
        let resp = http.get(url).send().await?;
        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !resp.status().is_success() {
            return Err(FetchError::Status(resp.status()));
        }
        Ok(Some(resp.bytes().await?))
    } else {
        Err(FetchError::UnsupportedScheme(url.to_owned()))
    }
}

async fn get_or_fetch(
    cache: &ResourceCache,
    http: &reqwest::Client,
    url: String,
) -> Result<CachedResource, Arc<FetchError>> {
    cache.try_get_with(url.clone(), fetch_resource(http, &url)).await
}

#[tokio::main]
async fn main() {
    let style = fixture_path("maplibre_demo.json");
    assert!(style.is_file(), "fixture style not found at {}", style.display());

    let resources: ResourceCache = Cache::builder()
        .max_capacity(2_000)
        .time_to_live(Duration::from_secs(3600))
        .build();

    // Connection pooling matters here: mbgl asks for many glyph PBFs
    // from the same host in quick succession.
    let http = reqwest::Client::builder()
        .user_agent("maplibre-native-rs/tile-server-example")
        .pool_max_idle_per_host(16)
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(15))
        .build()
        .expect("http client build");

    // Register BEFORE the pool's `LazyLock` is touched — the pool's
    // worker thread will then build its renderer with our factory
    // installed.
    register_file_source_callback(FileSourceRequestCallback::new_async(
        tokio::runtime::Handle::current(),
        {
            let resources = resources.clone();
            let http = http.clone();
            move |url, _| {
                let resources = resources.clone();
                let http = http.clone();
                async move {
                    match get_or_fetch(&resources, &http, url).await {
                        Ok(Some(b)) => FsResponse::Ok(b.to_vec()),
                        Ok(None) => FsResponse::NoContent,
                        Err(e) => FsResponse::Error {
                            reason: FsErrorReason::Other,
                            message: e.to_string(),
                        },
                    }
                }
            }
        },
    ));

    let warmup_urls = [
        format!("file://{}", style.display()),
        "https://demotiles.maplibre.org/tiles/tiles.json".to_owned(),
        // Latin-1 glyph ranges for the two fontstacks the demo style
        // uses; nearly every render needs these. Real deployments would
        // parse the style for its full set.
        "https://demotiles.maplibre.org/font/Open Sans Semibold,Arial Unicode MS Bold/0-255.pbf"
            .to_owned(),
        "https://demotiles.maplibre.org/font/Open Sans Regular,Arial Unicode MS Regular/0-255.pbf"
            .to_owned(),
    ];
    for url in warmup_urls {
        let resources = resources.clone();
        let http = http.clone();
        tokio::spawn(async move {
            let _ = get_or_fetch(&resources, &http, url).await;
        });
    }

    let state = AppState {
        rendered: Cache::builder()
            .max_capacity(1024)
            .time_to_live(Duration::from_secs(3600))
            .build(),
        style: Arc::new(style),
    };

    let addr = "127.0.0.1:3000";
    println!("Server running on http://{addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let app = Router::new()
        .route("/", get(index))
        .route("/{z}/{x}/{y}", get(rendered_style_tile))
        .with_state(state);
    axum::serve(listener, app).await.unwrap();
}

#[derive(Debug, thiserror::Error)]
enum RenderError {
    #[error("render pool: {0}")]
    Pool(#[from] maplibre_native::SingleThreadedRenderPoolError),
    #[error("png encode: {0}")]
    Encode(#[from] image::ImageError),
}

#[derive(Debug, thiserror::Error)]
enum FetchError {
    #[error("request: {0}")]
    Request(#[from] reqwest::Error),
    #[error("status: {0}")]
    Status(reqwest::StatusCode),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported scheme: {0}")]
    UnsupportedScheme(String),
}
