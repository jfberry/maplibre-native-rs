//! Rust-supplied FileSource callback.
//!
//! Pair with the C++ side defined in `src/cpp/rust_file_source.{h,cpp}`. The
//! Rust closure handed to
//! [`ImageRendererBuilder::with_file_source_callback`](crate::ImageRendererBuilder::with_file_source_callback)
//! (or its async sibling
//! [`ImageRendererBuilder::with_async_file_source_callback`](crate::ImageRendererBuilder::with_async_file_source_callback))
//! serves every resource mbgl asks for — styles, tilesets, tiles, glyphs,
//! sprites, images. The URL scheme is not filtered on the C++ side, so the
//! callback owns scheme dispatch (e.g. `mbtiles://`, `file://`, custom).

use std::fmt::Debug;

#[cfg(feature = "async-file-source")]
use std::{future::Future, pin::Pin};

/// Kind of resource being requested. Mirrors `mbgl::Resource::Kind` from
/// `mbgl/storage/resource.hpp`; discriminant values are pinned byte-for-byte
/// to that enum by `static_assert`s in `rust_file_source.cpp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ResourceKind {
    /// Unknown / unspecified resource kind.
    Unknown = 0,
    /// A style.json.
    Style = 1,
    /// A TileJSON / source descriptor.
    Source = 2,
    /// A single tile (vector or raster).
    Tile = 3,
    /// A glyph PBF range.
    Glyphs = 4,
    /// A sprite sheet PNG.
    SpriteImage = 5,
    /// A sprite sheet JSON.
    SpriteJSON = 6,
    /// A generic image resource.
    Image = 7,
}

impl ResourceKind {
    fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Style,
            2 => Self::Source,
            3 => Self::Tile,
            4 => Self::Glyphs,
            5 => Self::SpriteImage,
            6 => Self::SpriteJSON,
            7 => Self::Image,
            _ => Self::Unknown,
        }
    }
}

/// Error reason for a failed resource request. Mirrors
/// `mbgl::Response::Error::Reason`; discriminant values are pinned to that
/// enum by `static_assert`s in `rust_file_source.cpp`. The `0` discriminant
/// is intentionally reserved on the FFI side to mean "no error" —
/// `mbgl::Response::Error::Reason` starts at `Success = 1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FsErrorReason {
    /// Resource not found at the requested URL.
    NotFound = 2,
    /// Server-side error (5xx, etc.).
    Server = 3,
    /// Transport-level connection failure.
    Connection = 4,
    /// Rate-limit response.
    RateLimit = 5,
    /// Any other error.
    Other = 6,
}

/// Return value for a resource request callback.
#[derive(Debug)]
pub enum FsResponse {
    /// Request succeeded; bytes are the raw resource body.
    Ok(Vec<u8>),
    /// Request was a well-formed miss (e.g. mbtiles tile not present at
    /// this z/x/y). mbgl treats this as a 204-equivalent — no error, no
    /// body. Overzoomed tiles should use this rather than [`FsResponse::Error`].
    NoContent,
    /// Request failed. The reason maps directly to the mbgl error enum so
    /// mbgl-internal retry/backoff logic still applies.
    Error {
        /// Category of failure.
        reason: FsErrorReason,
        /// Human-readable message for logging.
        message: String,
    },
}

#[cfg(feature = "async-file-source")]
type AsyncFsFuture = Pin<Box<dyn Future<Output = FsResponse> + Send + 'static>>;

#[cfg(feature = "async-file-source")]
type SyncFn = Box<dyn Fn(&str, ResourceKind) -> FsResponse + Send + Sync + 'static>;
#[cfg(not(feature = "async-file-source"))]
type SyncFn = Box<dyn Fn(&str, ResourceKind) -> FsResponse + Send + Sync + 'static>;

#[cfg(feature = "async-file-source")]
type AsyncFn = Box<dyn Fn(String, ResourceKind) -> AsyncFsFuture + Send + Sync + 'static>;

/// Internal storage for the registered closure. Sync closures take the
/// inline dispatch path; async closures spawn on the configured tokio
/// runtime and deliver via `FsRequestSink::deliver` when the future resolves.
enum FsCallbackKind {
    Sync(SyncFn),
    #[cfg(feature = "async-file-source")]
    Async {
        callback: AsyncFn,
        runtime: tokio::runtime::Handle,
    },
}

impl Debug for FsCallbackKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sync(_) => write!(f, "FsCallbackKind::Sync"),
            #[cfg(feature = "async-file-source")]
            Self::Async { .. } => write!(f, "FsCallbackKind::Async"),
        }
    }
}

/// Opaque handle to a registered FileSource closure.
///
/// Construct via [`ImageRendererBuilder::with_file_source_callback`](crate::ImageRendererBuilder::with_file_source_callback)
/// or [`ImageRendererBuilder::with_async_file_source_callback`](crate::ImageRendererBuilder::with_async_file_source_callback).
/// Stored on the builder until `build_*_renderer` registers it with mbgl.
//
// `Send + Sync` are load-bearing: `mbgl::FileSourceManager` is a singleton,
// so the same callback is captured by every `RustFileSource` instance that
// the factory produces. If a consumer constructs more than one
// `ImageRenderer` (or mbgl ever spawns a second file-source-owning thread
// upstream), the callback will be invoked from multiple threads and must
// be thread-safe.
pub struct FileSourceRequestCallback {
    inner: FsCallbackKind,
}

impl Debug for FileSourceRequestCallback {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FileSourceRequestCallback").field("kind", &self.inner).finish()
    }
}

impl FileSourceRequestCallback {
    /// Wrap a sync closure. The closure is invoked inline on whatever
    /// thread mbgl issues the request from — fast for SQLite/filesystem
    /// backends, but slow callbacks will stall the render thread.
    pub fn new_sync<F>(callback: F) -> Self
    where
        F: Fn(&str, ResourceKind) -> FsResponse + Send + Sync + 'static,
    {
        Self { inner: FsCallbackKind::Sync(Box::new(callback)) }
    }

    /// Wrap an async closure that returns a `Future<Output = FsResponse>`.
    /// The future is spawned on `runtime`; on completion it delivers via
    /// the per-request sink. mbgl receives a cancellable handle — when
    /// it drops the handle, the response is discarded (the future itself
    /// runs to completion).
    #[cfg(feature = "async-file-source")]
    pub fn new_async<F, Fut>(runtime: tokio::runtime::Handle, callback: F) -> Self
    where
        F: Fn(String, ResourceKind) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = FsResponse> + Send + 'static,
    {
        let cb: AsyncFn = Box::new(move |url, kind| Box::pin(callback(url, kind)));
        Self { inner: FsCallbackKind::Async { callback: cb, runtime } }
    }
}

fn ffi_response(r: FsResponse) -> crate::renderer::bridge::file_source::RustFsResponse {
    use crate::renderer::bridge::file_source::RustFsResponse;
    match r {
        FsResponse::Ok(bytes) => RustFsResponse {
            data: bytes,
            error_reason: 0,
            error_message: String::new(),
            no_content: false,
        },
        FsResponse::NoContent => RustFsResponse {
            data: Vec::new(),
            error_reason: 0,
            error_message: String::new(),
            no_content: true,
        },
        FsResponse::Error { reason, message } => RustFsResponse {
            data: Vec::new(),
            error_reason: reason as u8,
            error_message: message,
            no_content: false,
        },
    }
}

/// Register a [`FileSourceRequestCallback`] as the process-global
/// `ResourceLoader` factory without going through the builder. Useful
/// for installing a callback ahead of [`SingleThreadedRenderPool`](crate::SingleThreadedRenderPool),
/// whose worker thread builds its renderer lazily.
///
/// See [`ImageRendererBuilder::with_file_source_callback`](crate::ImageRendererBuilder::with_file_source_callback)
/// for the singleton/threading caveats — they apply here too.
pub fn register_file_source_callback(callback: FileSourceRequestCallback) {
    crate::renderer::bridge::file_source::register_rust_file_source_factory(Box::new(callback));
}

/// Bridge predicate invoked by C++ to pick sync vs async dispatch.
pub(crate) fn fs_callback_is_sync(callback: &FileSourceRequestCallback) -> bool {
    match &callback.inner {
        FsCallbackKind::Sync(_) => true,
        #[cfg(feature = "async-file-source")]
        FsCallbackKind::Async { .. } => false,
    }
}

/// Bridge sync-dispatch entry point. Invoked by C++ inside
/// `RustFileSource::request` when `fs_callback_is_sync` returned true.
pub(crate) fn fs_invoke_sync(
    callback: &FileSourceRequestCallback,
    url: &str,
    kind: u8,
) -> crate::renderer::bridge::file_source::RustFsResponse {
    match &callback.inner {
        FsCallbackKind::Sync(f) => ffi_response(f(url, ResourceKind::from_u8(kind))),
        #[cfg(feature = "async-file-source")]
        FsCallbackKind::Async { .. } => {
            // The C++ side checks `fs_callback_is_sync` first; reaching
            // here means the dispatch logic on the C++ side has drifted.
            unreachable!("fs_invoke_sync called on async callback");
        }
    }
}

/// Bridge async-dispatch entry point. Spawns the closure's future on the
/// configured tokio runtime; the spawned task takes ownership of the
/// `FsRequestSink` and calls its `deliver` method when the future resolves.
#[cfg(feature = "async-file-source")]
pub(crate) fn fs_invoke_async(
    callback: &FileSourceRequestCallback,
    url: String,
    kind: u8,
    sink: cxx::UniquePtr<crate::renderer::bridge::file_source::FsRequestSink>,
) {
    match &callback.inner {
        FsCallbackKind::Async { callback: cb, runtime } => {
            let fut = cb(url, ResourceKind::from_u8(kind));
            // The sink owns the per-request shared state. The spawned task
            // moves it in; on drop after `deliver` (or without `deliver`,
            // if the future was dropped), the sink's destructor releases
            // its shared_ptr to the request state. The mbgl callback was
            // moved out under lock during `deliver`, so we never double-fire.
            runtime.spawn(async move {
                let response = fut.await;
                let mut sink = sink;
                sink.pin_mut().deliver(ffi_response(response));
            });
        }
        FsCallbackKind::Sync(_) => {
            unreachable!("fs_invoke_async called on sync callback");
        }
    }
}

/// Bridge stub when async support is compiled out. The C++ side never
/// invokes this because `fs_callback_is_sync` always returns true on
/// non-async builds — the dispatch path picks sync inline.
#[cfg(not(feature = "async-file-source"))]
pub(crate) fn fs_invoke_async(
    _callback: &FileSourceRequestCallback,
    _url: String,
    _kind: u8,
    _sink: cxx::UniquePtr<crate::renderer::bridge::file_source::FsRequestSink>,
) {
    unreachable!(
        "async file source callback dispatched without `async-file-source` feature; \
         enable the feature on the `maplibre_native` crate to use async callbacks"
    );
}
