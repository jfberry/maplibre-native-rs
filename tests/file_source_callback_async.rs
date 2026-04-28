//! End-to-end test for the async FileSource callback bridge.
//!
//! Same shape as the sync test, but the callback is `async fn`. The
//! future yields once via `tokio::task::yield_now` to ensure the spawn /
//! deliver path is exercised — a future that resolves immediately would
//! still flow through `cb()` synchronously because the runtime can poll
//! it to completion in one go. The yield forces a real cross-thread
//! deliver.
//!
//! Continuous-mode renderer: async deliveries land on tokio worker
//! threads, which mbgl tolerates in continuous mode. Static rendering
//! doesn't pump a run loop, so async + static is documented as not
//! recommended; we use a tile renderer here to stay clear of that.

#![cfg(feature = "async-file-source")]

use std::num::NonZeroU32;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use maplibre_native::{FsResponse, ImageRendererBuilder, ResourceKind};

const INLINE_STYLE: &str = r#"{
    "version": 8,
    "name": "callback-test",
    "sources": {},
    "layers": [
        { "id": "bg", "type": "background", "paint": { "background-color": "rgb(0, 200, 100)" } }
    ]
}"#;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn async_file_source_callback_serves_inline_style() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let call_count_cb = call_count.clone();

    let mut renderer = ImageRendererBuilder::new()
        .with_size(NonZeroU32::new(64).unwrap(), NonZeroU32::new(64).unwrap())
        .with_async_file_source_callback(move |url: String, _kind: ResourceKind| {
            let call_count_cb = call_count_cb.clone();
            async move {
                // Yield so the future doesn't resolve in one poll —
                // forces the spawned-task / sink-deliver path.
                tokio::task::yield_now().await;
                call_count_cb.fetch_add(1, Ordering::SeqCst);

                if url.ends_with("/inline-style.json") {
                    FsResponse::Ok(INLINE_STYLE.as_bytes().to_vec())
                } else {
                    FsResponse::NoContent
                }
            }
        })
        .build_tile_renderer();

    let url: url::Url = "https://example.invalid/inline-style.json".parse().unwrap();
    renderer.load_style_from_url(&url);

    // `render_tile` is a blocking C++ call. We're on a multi-threaded
    // runtime with worker_threads=2: this test's worker blocks inside
    // render_tile while the spawned async-callback task runs on the
    // other worker, fires `deliver`, and unblocks the render. The
    // renderer itself is `!Send`, so it has to stay on this thread —
    // we don't use `spawn_blocking`.
    let image = renderer.render_tile(0, 0, 0).expect("render should succeed");

    let buf = image.as_image();
    assert_eq!(buf.width(), 64);
    assert_eq!(buf.height(), 64);

    let calls = call_count.load(Ordering::SeqCst);
    assert!(calls >= 1, "expected ≥1 async callback invocation, got {calls}");
}
