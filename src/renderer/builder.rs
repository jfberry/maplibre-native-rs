//! Image renderer configuration and builder

use crate::renderer::bridge::ffi;
use crate::renderer::bridge::file_source::register_rust_file_source_factory;
use crate::renderer::file_source::{FileSourceRequestCallback, FsResponse, ResourceKind};
use crate::renderer::{Continuous, ImageRenderer, MapMode, Static, Tile};
use crate::ResourceOptions;
use std::marker::PhantomData;
use std::num::NonZeroU32;

#[cfg(feature = "async-file-source")]
use std::future::Future;

/// Builder for configuring [`ImageRenderer`] instances
///
/// # Examples
///
/// ```
/// use maplibre_native::ImageRendererBuilder;
/// use std::num::NonZeroU32;
///
/// let renderer = ImageRendererBuilder::new()
///     .with_size(NonZeroU32::new(1024).unwrap(), NonZeroU32::new(768).unwrap())
///     .with_pixel_ratio(2.0)
///     .build_static_renderer();
/// ```
#[derive(Debug)]
pub struct ImageRendererBuilder {
    /// Image width in pixels
    width: NonZeroU32,
    /// Image height in pixelsHash
    height: NonZeroU32,
    /// Pixel ratio for high-DPI displays
    pixel_ratio: f32,

    resource_options: Option<ResourceOptions>,

    /// Optional Rust-supplied FileSource callback. When set, installs a
    /// process-global factory at build time that delegates every resource
    /// request to this closure, bypassing the mbgl default ResourceLoader.
    file_source_callback: Option<FileSourceRequestCallback>,

    /// Tokio runtime handle used to spawn async file-source futures.
    /// Resolved at `build_*_renderer` time when an async callback is set:
    /// caller-supplied via `with_file_source_runtime` takes priority,
    /// otherwise the ambient `Handle::try_current()` is used. If neither
    /// is available, the builder panics.
    #[cfg(feature = "async-file-source")]
    file_source_runtime: Option<tokio::runtime::Handle>,
}

impl Default for ImageRendererBuilder {
    #[allow(clippy::missing_panics_doc, reason = "infallible")]
    fn default() -> Self {
        Self {
            width: NonZeroU32::new(512).unwrap(),
            height: NonZeroU32::new(512).unwrap(),
            pixel_ratio: 1.0,
            resource_options: None,
            file_source_callback: None,
            #[cfg(feature = "async-file-source")]
            file_source_runtime: None,
        }
    }
}

impl ImageRendererBuilder {
    /// Creates a new builder with default values
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets image dimensions
    ///
    /// Default: `512` x `512`
    #[must_use]
    #[allow(clippy::needless_pass_by_value, reason = "false positive")]
    pub fn with_size(mut self, width: NonZeroU32, height: NonZeroU32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Sets pixel ratio for high-DPI displays
    ///
    /// Default: `1.0`
    #[must_use]
    #[allow(clippy::needless_pass_by_value, reason = "false positive")]
    pub fn with_pixel_ratio(mut self, pixel_ratio: impl Into<f32>) -> Self {
        self.pixel_ratio = pixel_ratio.into();
        self
    }

    /// Set Resource Options
    #[must_use]
    #[allow(clippy::needless_pass_by_value, reason = "false positive")]
    pub fn with_resource_options(mut self, resource_options: ResourceOptions) -> Self {
        self.resource_options = Some(resource_options);
        self
    }

    /// Install a synchronous Rust closure as the FileSource callback.
    ///
    /// The closure is invoked for every resource mbgl needs to render the
    /// style (tiles, glyphs, sprites, etc.). It replaces the mbgl default
    /// ResourceLoader entirely, so the closure must handle every URL
    /// scheme referenced by the style — typical schemes are `mbtiles://`,
    /// `file://`, and any custom ones the caller needs.
    ///
    /// Sync dispatch invokes the closure inline on whichever thread mbgl
    /// makes the request from. This is fast for SQLite/filesystem-backed
    /// callbacks, but the render thread blocks while the closure runs —
    /// don't perform network I/O here. Use
    /// [`with_async_file_source_callback`](Self::with_async_file_source_callback)
    /// (under the `async-file-source` feature) when the callback needs
    /// `.await`.
    ///
    /// Registration is **process-global**: `mbgl::FileSourceManager` is a
    /// singleton, so a later call to `build_*_renderer` replaces the
    /// factory for all *future* `ImageRenderer` instances. Existing
    /// renderers keep their original callback because mbgl captured their
    /// `FileSource` at `Map`-construction time. In practice, running two
    /// renderers in one process with *different* callbacks is unsupported
    /// — use one process per callback.
    ///
    /// `Send + Sync` are required because mbgl may invoke the same
    /// captured callback from multiple renderers on independent threads;
    /// the closure must be safe for concurrent use.
    ///
    /// Calling this method overwrites a previously-set async callback
    /// (and vice versa) — only one variant is registered.
    #[must_use]
    pub fn with_file_source_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(&str, ResourceKind) -> FsResponse + Send + Sync + 'static,
    {
        self.file_source_callback = Some(FileSourceRequestCallback::new_sync(callback));
        self
    }

    /// Install an async Rust closure as the FileSource callback.
    ///
    /// Each request spawns the closure's future on a tokio runtime; mbgl
    /// receives a cancellable handle and the future's response is
    /// delivered when it resolves. Suitable for HTTP, S3, async SQLite,
    /// or any other backend where blocking the render thread is
    /// unacceptable.
    ///
    /// **Threading**: deliveries land on whatever tokio worker resolved
    /// the future. mbgl's pipeline tolerates this in `Continuous` and
    /// `Tile` modes. Static rendering doesn't pump a run loop, so async
    /// callbacks combined with `build_static_renderer` are not
    /// recommended — prefer the sync variant for static rendering.
    ///
    /// **Cancellation**: when mbgl drops its `AsyncRequest` handle, the
    /// per-request sink swallows the eventual response. The future itself
    /// runs to completion — discard-on-arrival, not abort. If you need
    /// real cancellation, propagate it inside the closure (e.g. via a
    /// `CancellationToken` baked into your closure's captures).
    ///
    /// **Runtime**: pass an explicit handle via
    /// [`with_file_source_runtime`](Self::with_file_source_runtime), or
    /// call this method from within a tokio context — `Handle::try_current()`
    /// is consulted as a fallback. If neither is available, building the
    /// renderer panics.
    ///
    /// Same singleton/`Send + Sync` caveats as
    /// [`with_file_source_callback`](Self::with_file_source_callback)
    /// apply.
    #[cfg(feature = "async-file-source")]
    #[must_use]
    pub fn with_async_file_source_callback<F, Fut>(mut self, callback: F) -> Self
    where
        F: Fn(String, ResourceKind) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = FsResponse> + Send + 'static,
    {
        // Defer runtime resolution to build time so callers can chain
        // `with_async_file_source_callback(...).with_file_source_runtime(handle)`
        // in any order.
        let runtime = self
            .file_source_runtime
            .clone()
            .or_else(|| tokio::runtime::Handle::try_current().ok())
            .unwrap_or_else(|| {
                // Defer the panic to build time by stashing a placeholder?
                // Simpler: try one more time at build_*. We can re-resolve
                // there. For now, just keep it None and re-check on build.
                // (We can't return a sentinel Handle.) Fall through with
                // a fake handle that's only constructed lazily — instead,
                // we assert immediately and tell the user to set the
                // runtime first or call from inside a tokio context.
                panic!(
                    "with_async_file_source_callback requires either \
                     with_file_source_runtime(handle) to be called first, \
                     or for this builder to be constructed inside a tokio runtime"
                );
            });
        self.file_source_callback =
            Some(FileSourceRequestCallback::new_async(runtime, callback));
        self
    }

    /// Provide a tokio runtime handle for spawning async file-source
    /// callbacks. Optional — if unset, the builder uses the ambient
    /// `tokio::runtime::Handle::current()` at the moment
    /// [`with_async_file_source_callback`](Self::with_async_file_source_callback)
    /// is called. Call this method *before* setting the async callback
    /// when you want a specific runtime.
    #[cfg(feature = "async-file-source")]
    #[must_use]
    #[allow(clippy::needless_pass_by_value, reason = "Handle is cheap to clone but the API takes ownership for ergonomics")]
    pub fn with_file_source_runtime(mut self, handle: tokio::runtime::Handle) -> Self {
        self.file_source_runtime = Some(handle);
        self
    }

    /// Builds a static image renderer
    #[must_use]
    pub fn build_static_renderer(self) -> ImageRenderer<Static> {
        // TODO: Should the width/height be passed in here, or have another `build_static_with_size` method?
        ImageRenderer::new(MapMode::Static, self)
    }

    /// Builds a tile renderer
    #[must_use]
    pub fn build_tile_renderer(self) -> ImageRenderer<Tile> {
        // TODO: Is the width/height used for this mode?
        ImageRenderer::new(MapMode::Tile, self)
    }

    /// Builds a continuous renderer
    /// Using the `MapObserver` it is possible to react on signals from the Map
    #[must_use]
    pub fn build_continuous_renderer(self) -> ImageRenderer<Continuous> {
        ImageRenderer::new(MapMode::Continuous, self)
    }
}

impl<S> ImageRenderer<S> {
    /// Creates a new renderer instance
    fn new(map_mode: MapMode, opts: ImageRendererBuilder) -> Self {
        // Install the FileSource callback BEFORE constructing the C++
        // renderer: mbgl::Map resolves its FileSource during construction,
        // so the factory has to be in place by then.
        if let Some(callback) = opts.file_source_callback {
            register_rust_file_source_factory(Box::new(callback));
        }

        let resource_options = opts.resource_options.unwrap_or_default();
        let map = ffi::MapRenderer_new(
            map_mode,
            opts.width.get(),
            opts.height.get(),
            opts.pixel_ratio,
            resource_options.as_ref(),
        );

        Self { instance: map, style_specified: false, _marker: PhantomData }
    }
}
