#pragma once

// Rust-backed FileSource bridge.
//
// Installs an `mbgl::FileSource` factory for `FileSourceType::ResourceLoader`
// that delegates every resource request to a Rust closure. This replaces the
// default ResourceLoader (which composes Asset/Database/Network/Mbtiles/Pmtiles
// sources) with a single Rust-supplied handler — letting callers serve
// mbtiles://, file://, and custom schemes from Rust without running a sidecar
// HTTP server or pre-extracting tiles.
//
// Factory registration is process-global (mbgl::FileSourceManager is a
// singleton). Call `register_rust_file_source_factory` once before any
// `mbgl::Map` is constructed. A subsequent call replaces the previous
// callback but leaves existing RustFileSource instances alive until their
// owning Map is destroyed.
//
// Two dispatch modes:
//
//  - Sync: `RustFileSource::request` invokes the Rust closure inline and
//    delivers the response before returning. Returns a `NoopAsyncRequest`
//    because cancellation is structurally a no-op.
//
//  - Async: `RustFileSource::request` builds an `FsRequestSink` (holding
//    the mbgl `Callback` plus a cancellation flag), hands it to Rust which
//    spawns a tokio task, and returns a `RustAsyncRequest`. When mbgl drops
//    the `RustAsyncRequest`, the sink's `cancelled` flag is set; if the
//    Rust task delivers later, the sink swallows the response.

#include "rust/cxx.h"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>

#include <mbgl/storage/response.hpp>

namespace mbgl {
class FileSource;  // forward, for `Callback`'s function signature
}

namespace mln {
namespace bridge {

// Opaque Rust types — defined in src/renderer/file_source.rs.
struct FileSourceRequestCallback;
struct RustFsResponse;

// Shared state between an in-flight request's `RustAsyncRequest` (held by
// mbgl as the cancellation handle) and its `FsRequestSink` (held by the
// spawned Rust task). Owned by `shared_ptr` on both sides.
struct FsRequestShared {
    std::mutex mu;
    std::atomic<bool> cancelled{false};
    // mbgl's per-request callback. Cleared after delivery or on cancellation.
    std::function<void(mbgl::Response)> cb;
};

// Sink object that the Rust task uses to deliver the response. Crossed
// across the cxx boundary as a `UniquePtr<FsRequestSink>` so the spawned
// future owns it and drops it when done.
class FsRequestSink {
public:
    explicit FsRequestSink(std::shared_ptr<FsRequestShared> shared) noexcept
        : shared_(std::move(shared)) {}

    // Called once from the Rust async task once its future resolves.
    // No-op if the request has been cancelled in the meantime.
    void deliver(RustFsResponse response) noexcept;

private:
    std::shared_ptr<FsRequestShared> shared_;
};

// Implementation in rust_file_source.cpp.
void register_rust_file_source_factory(rust::Box<FileSourceRequestCallback> callback);

} // namespace bridge
} // namespace mln
