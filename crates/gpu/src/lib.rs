// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! `astral-gpu`: the single `unsafe` boundary for GPU access (R-RT-1).
//!
//! `libcuda` and NVML are loaded at runtime, so Astral binaries build and start
//! on machines without CUDA and report a clear error instead of failing to link.
//! Every handle here owns an `Arc<Context>`, so a context always outlives the
//! buffers, streams, events and modules created in it.

/// Generates a table of `extern "C"` function pointers loaded by symbol name.
macro_rules! api_table {
    ($vis:vis struct $tname:ident -> $ret:ty { $( $name:ident ( $($arg:ty),* ) ; )* }) => {
        #[allow(non_snake_case)]
        $vis struct $tname {
            $( pub $name: unsafe extern "C" fn($($arg),*) -> $ret, )*
        }
        impl $tname {
            #[allow(non_snake_case)]
            pub(crate) unsafe fn load(lib: &libloading::Library) -> std::result::Result<Self, String> {
                $(
                    // SAFETY: the caller guarantees each symbol has the declared signature.
                    let $name = *unsafe { lib.get::<unsafe extern "C" fn($($arg),*) -> $ret>(concat!(stringify!($name), "\0").as_bytes()) }
                        .map_err(|e| format!("{}: {}", stringify!($name), e))?;
                )*
                Ok(Self { $($name,)* })
            }
        }
    };
}

pub mod graph;
pub mod nvml;
pub mod sys;
pub mod vmm;

pub use graph::GraphExec;
pub use vmm::VirtualRange;

use astral_core::{AstralError, Result};
use std::ffi::{c_char, c_void, CStr, CString};
use std::marker::PhantomData;
use std::sync::Arc;

/// A raw device address.
pub type DevicePtr = sys::CUdeviceptr;

fn drv() -> Result<&'static sys::Driver> {
    sys::driver().map_err(|detail| AstralError::LibraryNotFound { lib: "libcuda".into(), detail })
}

fn check(call: &'static str, code: sys::CUresult) -> Result<()> {
    if code == sys::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(AstralError::Cuda { call, code, name: error_name(code) })
    }
}

fn error_name(code: sys::CUresult) -> String {
    if let Ok(d) = sys::driver() {
        let mut p: *const c_char = std::ptr::null();
        // SAFETY: cuGetErrorName writes a pointer to a static string or fails.
        unsafe {
            if (d.fns.cuGetErrorName)(code, &mut p) == sys::CUDA_SUCCESS && !p.is_null() {
                return CStr::from_ptr(p).to_string_lossy().into_owned();
            }
        }
    }
    format!("CUresult {code}")
}

/// Whether the CUDA driver can be loaded and initialised on this machine.
pub fn is_available() -> bool {
    sys::driver().is_ok()
}

/// CUDA driver API version as reported by `cuDriverGetVersion` (e.g. 13000).
/// Starts (`true`) or stops collection for an attached profiler (Nsight); a no-op otherwise.
pub fn profiler(on: bool) -> Result<()> {
    let d = drv()?;
    // SAFETY: no arguments; valid at any time after cuInit.
    check(if on { "cuProfilerStart" } else { "cuProfilerStop" }, unsafe { if on { (d.fns.cuProfilerStart)() } else { (d.fns.cuProfilerStop)() } })
}

pub fn driver_version() -> Result<i32> {
    let d = drv()?;
    let mut v = 0;
    // SAFETY: out-pointer to a local.
    check("cuDriverGetVersion", unsafe { (d.fns.cuDriverGetVersion)(&mut v) })?;
    Ok(v)
}

/// `CUdevice_attribute` values used by Astral (from `cuda.h`).
pub mod attr {
    pub const MAX_THREADS_PER_BLOCK: i32 = 1;
    pub const WARP_SIZE: i32 = 10;
    pub const CLOCK_RATE_KHZ: i32 = 13;
    pub const MULTIPROCESSOR_COUNT: i32 = 16;
    pub const INTEGRATED: i32 = 18;
    pub const ECC_ENABLED: i32 = 32;
    pub const PCI_BUS_ID: i32 = 33;
    pub const PCI_DEVICE_ID: i32 = 34;
    pub const MEMORY_CLOCK_RATE_KHZ: i32 = 36;
    pub const GLOBAL_MEMORY_BUS_WIDTH: i32 = 37;
    pub const L2_CACHE_SIZE: i32 = 38;
    pub const MAX_THREADS_PER_MULTIPROCESSOR: i32 = 39;
    pub const PCI_DOMAIN_ID: i32 = 50;
    pub const COMPUTE_CAPABILITY_MAJOR: i32 = 75;
    pub const COMPUTE_CAPABILITY_MINOR: i32 = 76;
    pub const MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: i32 = 81;
    pub const MAX_REGISTERS_PER_MULTIPROCESSOR: i32 = 82;
    pub const MANAGED_MEMORY: i32 = 83;
    pub const PAGEABLE_MEMORY_ACCESS: i32 = 88;
    pub const CONCURRENT_MANAGED_ACCESS: i32 = 89;
    pub const MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: i32 = 97;
    pub const VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED: i32 = 102;
    pub const MEMORY_POOLS_SUPPORTED: i32 = 115;
    pub const CLUSTER_LAUNCH: i32 = 120;
}

/// A CUDA device handle.
#[derive(Clone, Copy, Debug)]
pub struct Device {
    ordinal: i32,
    raw: sys::CUdevice,
}

impl Device {
    pub fn count() -> Result<usize> {
        let d = drv()?;
        let mut n = 0;
        // SAFETY: out-pointer to a local.
        check("cuDeviceGetCount", unsafe { (d.fns.cuDeviceGetCount)(&mut n) })?;
        Ok(n.max(0) as usize)
    }

    pub fn get(ordinal: usize) -> Result<Self> {
        let d = drv()?;
        let mut raw = 0;
        // SAFETY: out-pointer to a local.
        check("cuDeviceGet", unsafe { (d.fns.cuDeviceGet)(&mut raw, ordinal as i32) })?;
        Ok(Self { ordinal: ordinal as i32, raw })
    }

    pub fn ordinal(&self) -> usize {
        self.ordinal as usize
    }

    pub fn name(&self) -> Result<String> {
        let d = drv()?;
        let mut buf = [0 as c_char; 256];
        // SAFETY: buffer length is passed; the driver NUL-terminates within it.
        check("cuDeviceGetName", unsafe { (d.fns.cuDeviceGetName)(buf.as_mut_ptr(), buf.len() as i32, self.raw) })?;
        // SAFETY: NUL-terminated by the driver.
        Ok(unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned())
    }

    /// UUID formatted like NVML/nvidia-smi: `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`.
    pub fn uuid(&self) -> Result<String> {
        let d = drv()?;
        let mut u = [0u8; 16];
        // SAFETY: CUuuid is 16 bytes.
        check("cuDeviceGetUuid", unsafe { (d.fns.cuDeviceGetUuid)(u.as_mut_ptr(), self.raw) })?;
        let h = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
        Ok(format!("GPU-{}-{}-{}-{}-{}", h(&u[0..4]), h(&u[4..6]), h(&u[6..8]), h(&u[8..10]), h(&u[10..16])))
    }

    pub fn total_mem(&self) -> Result<usize> {
        let d = drv()?;
        let mut b = 0usize;
        // SAFETY: out-pointer to a local.
        check("cuDeviceTotalMem_v2", unsafe { (d.fns.cuDeviceTotalMem_v2)(&mut b, self.raw) })?;
        Ok(b)
    }

    pub fn attribute(&self, attribute: i32) -> Result<i32> {
        let d = drv()?;
        let mut v = 0;
        // SAFETY: out-pointer to a local.
        check("cuDeviceGetAttribute", unsafe { (d.fns.cuDeviceGetAttribute)(&mut v, attribute, self.raw) })?;
        Ok(v)
    }

    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        Ok((self.attribute(attr::COMPUTE_CAPABILITY_MAJOR)?, self.attribute(attr::COMPUTE_CAPABILITY_MINOR)?))
    }
}

/// The device's primary context. Released when the last `Arc` is dropped.
pub struct Context {
    device: Device,
    raw: sys::CUcontext,
    /// Launch kernels with programmatic dependent launch (Hopper+; kernels begin with
    /// `griddepcontrol.wait`, so this only overlaps launch latency, never data). `pdl_supported`
    /// is fixed; `pdl` can be switched off per phase (e.g. prefill).
    pdl_supported: bool,
    pdl: std::sync::atomic::AtomicBool,
}

// SAFETY: a primary-context handle may be made current and used from any thread.
unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Context {
    /// Retains the device's primary context and makes it current on this thread.
    pub fn new(device: Device) -> Result<Arc<Self>> {
        let d = drv()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: out-pointer to a local; released in Drop.
        check("cuDevicePrimaryCtxRetain", unsafe { (d.fns.cuDevicePrimaryCtxRetain)(&mut raw, device.raw) })?;
        let pdl = device.compute_capability().map(|(m, _)| m >= 9).unwrap_or(false) && std::env::var("ASTRAL_PDL").map_or(true, |v| v != "0");
        let ctx = Arc::new(Self { device, raw, pdl_supported: pdl, pdl: std::sync::atomic::AtomicBool::new(pdl) });
        ctx.bind()?;
        Ok(ctx)
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Whether kernels are launched with programmatic dependent launch.
    pub fn pdl(&self) -> bool {
        self.pdl.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Enables or disables programmatic dependent launch for later launches (and captures);
    /// returns the previous setting. No effect where it is unsupported.
    pub fn set_pdl(&self, on: bool) -> bool {
        self.pdl.swap(on && self.pdl_supported, std::sync::atomic::Ordering::Relaxed)
    }

    /// Makes this context current on the calling thread.
    pub fn bind(&self) -> Result<()> {
        let d = drv()?;
        // SAFETY: `raw` is a retained primary context.
        check("cuCtxSetCurrent", unsafe { (d.fns.cuCtxSetCurrent)(self.raw) })
    }

    pub fn synchronize(&self) -> Result<()> {
        self.bind()?;
        let d = drv()?;
        // SAFETY: context is current.
        check("cuCtxSynchronize", unsafe { (d.fns.cuCtxSynchronize)() })
    }

    /// (free, total) bytes as reported by `cuMemGetInfo`.
    /// On unified-memory devices this over-reports free memory (spec 01 §4).
    pub fn mem_info(&self) -> Result<(usize, usize)> {
        self.bind()?;
        let d = drv()?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: out-pointers to locals.
        check("cuMemGetInfo_v2", unsafe { (d.fns.cuMemGetInfo_v2)(&mut free, &mut total) })?;
        Ok((free, total))
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if let Ok(d) = sys::driver() {
            // SAFETY: balances the retain in `new`.
            unsafe {
                (d.fns.cuDevicePrimaryCtxRelease_v2)(self.device.raw);
            }
        }
    }
}

/// Types that may be copied bit-for-bit between host and device.
///
/// # Safety
/// Implementors must be plain old data: no pointers, no invariants on bit patterns.
pub unsafe trait DeviceCopy: Copy + Send + Sync + 'static {}
unsafe impl DeviceCopy for u8 {}
unsafe impl DeviceCopy for i8 {}
unsafe impl DeviceCopy for u16 {}
unsafe impl DeviceCopy for i16 {}
unsafe impl DeviceCopy for u32 {}
unsafe impl DeviceCopy for i32 {}
unsafe impl DeviceCopy for u64 {}
unsafe impl DeviceCopy for i64 {}
unsafe impl DeviceCopy for f32 {}
unsafe impl DeviceCopy for f64 {}

/// A typed device allocation of `len` elements.
pub struct DeviceBuffer<T: DeviceCopy> {
    ptr: sys::CUdeviceptr,
    len: usize,
    ctx: Arc<Context>,
    _t: PhantomData<T>,
}

impl<T: DeviceCopy> DeviceBuffer<T> {
    /// Allocates `len` elements and zero-fills them.
    pub fn zeroed(ctx: &Arc<Context>, len: usize) -> Result<Self> {
        let bytes = len.checked_mul(size_of::<T>()).ok_or_else(|| AstralError::Invalid("allocation size overflows usize".into()))?;
        ctx.bind()?;
        let d = drv()?;
        let mut ptr = 0;
        if bytes > 0 {
            // SAFETY: out-pointer to a local; freed in Drop.
            check("cuMemAlloc_v2", unsafe { (d.fns.cuMemAlloc_v2)(&mut ptr, bytes) })?;
        }
        let buf = Self { ptr, len, ctx: ctx.clone(), _t: PhantomData };
        if bytes > 0 {
            // SAFETY: `ptr` is a live allocation of `bytes` bytes.
            check("cuMemsetD8_v2", unsafe { (d.fns.cuMemsetD8_v2)(buf.ptr, 0, bytes) })?;
        }
        Ok(buf)
    }

    pub fn from_slice(ctx: &Arc<Context>, data: &[T]) -> Result<Self> {
        let mut b = Self::zeroed(ctx, data.len())?;
        b.copy_from_slice(data)?;
        Ok(b)
    }

    /// Synchronous host-to-device copy. `data.len()` must equal `self.len()`.
    pub fn copy_from_slice(&mut self, data: &[T]) -> Result<()> {
        if data.len() != self.len {
            return Err(AstralError::Invalid(format!("copy_from_slice: {} elements into buffer of {}", data.len(), self.len)));
        }
        if self.bytes() == 0 {
            return Ok(());
        }
        self.ctx.bind()?;
        let d = drv()?;
        // SAFETY: both regions are `bytes()` long; the host slice is valid for reads.
        check("cuMemcpyHtoD_v2", unsafe { (d.fns.cuMemcpyHtoD_v2)(self.ptr, data.as_ptr() as *const c_void, self.bytes()) })
    }

    /// Synchronous device-to-host copy of the whole buffer.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        let mut v: Vec<T> = Vec::with_capacity(self.len);
        if self.bytes() > 0 {
            self.ctx.bind()?;
            let d = drv()?;
            // SAFETY: `v` has capacity for `len` elements; length is set only after success.
            unsafe {
                check("cuMemcpyDtoH_v2", (d.fns.cuMemcpyDtoH_v2)(v.as_mut_ptr() as *mut c_void, self.ptr, self.bytes()))?;
                v.set_len(self.len);
            }
        }
        Ok(v)
    }

    /// Asynchronous device-to-device copy on `stream`. Lengths must match.
    pub fn copy_from_device_async(&mut self, src: &DeviceBuffer<T>, stream: &Stream) -> Result<()> {
        if src.len != self.len {
            return Err(AstralError::Invalid("copy_from_device_async: length mismatch".into()));
        }
        if self.bytes() == 0 {
            return Ok(());
        }
        self.ctx.bind()?;
        let d = drv()?;
        // SAFETY: both allocations are live and `bytes()` long; ordering is on `stream`.
        check("cuMemcpyDtoDAsync_v2", unsafe { (d.fns.cuMemcpyDtoDAsync_v2)(self.ptr, src.ptr, self.bytes(), stream.raw) })
    }

    /// Asynchronously zero-fills the buffer on `stream`.
    pub fn zero_async(&mut self, stream: &Stream) -> Result<()> {
        if self.bytes() == 0 {
            return Ok(());
        }
        self.ctx.bind()?;
        let d = drv()?;
        // SAFETY: live allocation of `bytes()` bytes; ordering is on `stream`.
        check("cuMemsetD8Async", unsafe { (d.fns.cuMemsetD8Async)(self.ptr, 0, self.bytes(), stream.raw) })
    }

    /// Asynchronously zero-fills `len` elements from `offset` on `stream`.
    pub fn zero_range_async(&mut self, offset: usize, len: usize, stream: &Stream) -> Result<()> {
        let p = self.range(offset, len)?;
        if len == 0 {
            return Ok(());
        }
        self.ctx.bind()?;
        let d = drv()?;
        // SAFETY: the range was bounds-checked against the live allocation; ordering is on `stream`.
        check("cuMemsetD8Async", unsafe { (d.fns.cuMemsetD8Async)(p, 0, len * size_of::<T>(), stream.raw) })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn bytes(&self) -> usize {
        self.len * size_of::<T>()
    }

    pub fn device_ptr(&self) -> DevicePtr {
        self.ptr
    }

    pub fn context(&self) -> &Arc<Context> {
        &self.ctx
    }
}

impl<T: DeviceCopy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.ptr != 0
            && let Ok(d) = sys::driver()
        {
            let _ = self.ctx.bind();
            // SAFETY: `ptr` came from cuMemAlloc and is freed exactly once.
            unsafe {
                (d.fns.cuMemFree_v2)(self.ptr);
            }
        }
    }
}

/// Read access to a typed device range (a whole buffer or a view of one).
pub trait Dev<T: DeviceCopy> {
    fn dptr(&self) -> DevicePtr;
    fn dlen(&self) -> usize;
}

/// Write access to a typed device range.
pub trait DevMut<T: DeviceCopy>: Dev<T> {}

impl<T: DeviceCopy> Dev<T> for DeviceBuffer<T> {
    fn dptr(&self) -> DevicePtr {
        self.ptr
    }
    fn dlen(&self) -> usize {
        self.len
    }
}
impl<T: DeviceCopy> DevMut<T> for DeviceBuffer<T> {}

/// A shared borrow of `len` elements of a [`DeviceBuffer`], starting at an element offset.
pub struct DeviceView<'a, T: DeviceCopy> {
    ptr: DevicePtr,
    len: usize,
    _b: PhantomData<&'a DeviceBuffer<T>>,
}

/// An exclusive borrow of a range of a [`DeviceBuffer`].
pub struct DeviceViewMut<'a, T: DeviceCopy> {
    ptr: DevicePtr,
    len: usize,
    _b: PhantomData<&'a mut DeviceBuffer<T>>,
}

impl<T: DeviceCopy> Dev<T> for DeviceView<'_, T> {
    fn dptr(&self) -> DevicePtr {
        self.ptr
    }
    fn dlen(&self) -> usize {
        self.len
    }
}
impl<T: DeviceCopy> Dev<T> for DeviceViewMut<'_, T> {
    fn dptr(&self) -> DevicePtr {
        self.ptr
    }
    fn dlen(&self) -> usize {
        self.len
    }
}
impl<T: DeviceCopy> DevMut<T> for DeviceViewMut<'_, T> {}

impl<'a, T: DeviceCopy> DeviceViewMut<'a, T> {
    fn range(&self, offset: usize, len: usize) -> Result<DevicePtr> {
        match offset.checked_add(len) {
            Some(end) if end <= self.len => Ok(self.ptr + (offset * size_of::<T>()) as u64),
            _ => Err(AstralError::Invalid(format!("view [{offset}, +{len}) outside view of {}", self.len))),
        }
    }

    /// A shared sub-view `[offset, offset + len)` of this view.
    pub fn view(&self, offset: usize, len: usize) -> Result<DeviceView<'_, T>> {
        Ok(DeviceView { ptr: self.range(offset, len)?, len, _b: PhantomData })
    }

    /// An exclusive sub-view `[offset, offset + len)` of this view.
    pub fn view_mut(&mut self, offset: usize, len: usize) -> Result<DeviceViewMut<'_, T>> {
        Ok(DeviceViewMut { ptr: self.range(offset, len)?, len, _b: PhantomData })
    }

    /// Splits this view into two disjoint views: `[0, mid)` and `[mid, len)`.
    pub fn split_at_mut_views(self, mid: usize) -> Result<(DeviceViewMut<'a, T>, DeviceViewMut<'a, T>)> {
        if mid > self.len {
            return Err(AstralError::Invalid(format!("split at {mid} outside view of {}", self.len)));
        }
        let b = self.ptr + (mid * size_of::<T>()) as u64;
        Ok((DeviceViewMut { ptr: self.ptr, len: mid, _b: PhantomData }, DeviceViewMut { ptr: b, len: self.len - mid, _b: PhantomData }))
    }
}

impl<T: DeviceCopy> DeviceBuffer<T> {
    fn range(&self, offset: usize, len: usize) -> Result<DevicePtr> {
        match offset.checked_add(len) {
            Some(end) if end <= self.len => Ok(self.ptr + (offset * size_of::<T>()) as u64),
            _ => Err(AstralError::Invalid(format!("view [{offset}, +{len}) outside buffer of {}", self.len))),
        }
    }

    pub fn view(&self, offset: usize, len: usize) -> Result<DeviceView<'_, T>> {
        Ok(DeviceView { ptr: self.range(offset, len)?, len, _b: PhantomData })
    }

    pub fn view_mut(&mut self, offset: usize, len: usize) -> Result<DeviceViewMut<'_, T>> {
        Ok(DeviceViewMut { ptr: self.range(offset, len)?, len, _b: PhantomData })
    }

    /// Two disjoint mutable views: `[0, mid)` and `[mid, len)`.
    pub fn split_at_mut_views(&mut self, mid: usize) -> Result<(DeviceViewMut<'_, T>, DeviceViewMut<'_, T>)> {
        if mid > self.len {
            return Err(AstralError::Invalid(format!("split at {mid} outside buffer of {}", self.len)));
        }
        let b = self.ptr + (mid * size_of::<T>()) as u64;
        Ok((DeviceViewMut { ptr: self.ptr, len: mid, _b: PhantomData }, DeviceViewMut { ptr: b, len: self.len - mid, _b: PhantomData }))
    }

    /// Asynchronous device-to-device copy of `len` elements between ranges of two buffers.
    pub fn copy_range_from_async(&mut self, dst_off: usize, src: &DeviceBuffer<T>, src_off: usize, len: usize, stream: &Stream) -> Result<()> {
        let dst = self.range(dst_off, len)?;
        let s = src.range(src_off, len)?;
        if len == 0 {
            return Ok(());
        }
        self.ctx.bind()?;
        let d = drv()?;
        // SAFETY: both ranges were bounds-checked; ordering is on `stream`.
        check("cuMemcpyDtoDAsync_v2", unsafe { (d.fns.cuMemcpyDtoDAsync_v2)(dst, s, len * size_of::<T>(), stream.raw) })
    }
}

/// A non-blocking CUDA stream.
pub struct Stream {
    raw: sys::CUstream,
    ctx: Arc<Context>,
}

// SAFETY: stream handles may be used from any thread with the context current.
unsafe impl Send for Stream {}

impl Stream {
    pub fn new(ctx: &Arc<Context>) -> Result<Self> {
        ctx.bind()?;
        let d = drv()?;
        let mut raw = std::ptr::null_mut();
        const CU_STREAM_NON_BLOCKING: u32 = 0x1;
        // SAFETY: out-pointer to a local; destroyed in Drop.
        check("cuStreamCreate", unsafe { (d.fns.cuStreamCreate)(&mut raw, CU_STREAM_NON_BLOCKING) })?;
        Ok(Self { raw, ctx: ctx.clone() })
    }

    pub fn synchronize(&self) -> Result<()> {
        self.ctx.bind()?;
        let d = drv()?;
        // SAFETY: live stream.
        check("cuStreamSynchronize", unsafe { (d.fns.cuStreamSynchronize)(self.raw) })
    }

    pub fn context(&self) -> &Arc<Context> {
        &self.ctx
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Ok(d) = sys::driver() {
            let _ = self.ctx.bind();
            // SAFETY: destroyed exactly once.
            unsafe {
                (d.fns.cuStreamDestroy_v2)(self.raw);
            }
        }
    }
}

/// A timing-capable CUDA event.
pub struct Event {
    raw: sys::CUevent,
    ctx: Arc<Context>,
}

// SAFETY: event handles may be used from any thread with the context current.
unsafe impl Send for Event {}

impl Event {
    pub fn new(ctx: &Arc<Context>) -> Result<Self> {
        ctx.bind()?;
        let d = drv()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: out-pointer to a local; flags 0 = default (timing enabled).
        check("cuEventCreate", unsafe { (d.fns.cuEventCreate)(&mut raw, 0) })?;
        Ok(Self { raw, ctx: ctx.clone() })
    }

    pub fn record(&self, stream: &Stream) -> Result<()> {
        self.ctx.bind()?;
        let d = drv()?;
        // SAFETY: live event and stream in the same context.
        check("cuEventRecord", unsafe { (d.fns.cuEventRecord)(self.raw, stream.raw) })
    }

    pub fn synchronize(&self) -> Result<()> {
        self.ctx.bind()?;
        let d = drv()?;
        // SAFETY: live event.
        check("cuEventSynchronize", unsafe { (d.fns.cuEventSynchronize)(self.raw) })
    }

    /// Milliseconds from `self` to `end`. Both must have completed.
    pub fn elapsed_ms(&self, end: &Event) -> Result<f32> {
        self.ctx.bind()?;
        let d = drv()?;
        let mut ms = 0f32;
        // SAFETY: out-pointer to a local; both events live.
        check("cuEventElapsedTime", unsafe { (d.cuEventElapsedTime)(&mut ms, self.raw, end.raw) })?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if let Ok(d) = sys::driver() {
            let _ = self.ctx.bind();
            // SAFETY: destroyed exactly once.
            unsafe {
                (d.fns.cuEventDestroy_v2)(self.raw);
            }
        }
    }
}

/// A loaded cubin/fatbin module.
pub struct Module {
    raw: sys::CUmodule,
    ctx: Arc<Context>,
}

// SAFETY: module handles are context-scoped and immutable after load.
unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Module {
    /// Loads a cubin or fatbin image. The image is copied to 8-byte-aligned storage first.
    pub fn load(ctx: &Arc<Context>, image: &[u8]) -> Result<Arc<Self>> {
        if image.is_empty() {
            return Err(AstralError::Invalid("empty kernel image".into()));
        }
        let mut aligned = vec![0u64; image.len().div_ceil(8) + 1];
        // SAFETY: `aligned` holds at least `image.len()` bytes; regions do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(image.as_ptr(), aligned.as_mut_ptr() as *mut u8, image.len());
        }
        ctx.bind()?;
        let d = drv()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: the driver reads the image during this call only.
        check("cuModuleLoadData", unsafe { (d.fns.cuModuleLoadData)(&mut raw, aligned.as_ptr() as *const c_void) })?;
        Ok(Arc::new(Self { raw, ctx: ctx.clone() }))
    }

    pub fn function(self: &Arc<Self>, name: &str) -> Result<Function> {
        let cname = CString::new(name).map_err(|_| AstralError::Invalid(format!("kernel name {name:?}")))?;
        self.ctx.bind()?;
        let d = drv()?;
        let mut raw = std::ptr::null_mut();
        // SAFETY: out-pointer to a local; `cname` outlives the call.
        check("cuModuleGetFunction", unsafe { (d.fns.cuModuleGetFunction)(&mut raw, self.raw, cname.as_ptr()) }).map_err(|e| match e {
            AstralError::Cuda { code, name: n, .. } => AstralError::Cuda { call: "cuModuleGetFunction", code, name: format!("{n} (kernel `{name}`)") },
            other => other,
        })?;
        Ok(Function { raw, module: self.clone() })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if let Ok(d) = sys::driver() {
            let _ = self.ctx.bind();
            // SAFETY: unloaded exactly once; Functions hold an Arc to keep it alive.
            unsafe {
                (d.fns.cuModuleUnload)(self.raw);
            }
        }
    }
}

/// A kernel entry point. Keeps its module loaded.
pub struct Function {
    raw: sys::CUfunction,
    module: Arc<Module>,
}

// SAFETY: function handles are immutable and context-scoped.
unsafe impl Send for Function {}
unsafe impl Sync for Function {}

/// Launch geometry.
#[derive(Clone, Copy, Debug)]
pub struct LaunchConfig {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

impl Function {
    /// Allows launches with up to `bytes` of dynamic shared memory (above the 48 KB default;
    /// `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES`).
    pub fn set_max_dynamic_shared(&self, bytes: u32) -> Result<()> {
        self.module.ctx.bind()?;
        let d = drv()?;
        const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;
        // SAFETY: live function handle; the attribute only changes launch limits.
        check("cuFuncSetAttribute", unsafe { (d.fns.cuFuncSetAttribute)(self.raw, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes as i32) })
    }

    /// Preferred shared-memory carveout in percent of the maximum (0 = prefer L1, 100 = prefer
    /// shared; `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT`). A hint: a kernel that needs
    /// more shared memory still gets it.
    pub fn set_carveout(&self, percent: i32) -> Result<()> {
        self.module.ctx.bind()?;
        let d = drv()?;
        const CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT: i32 = 9;
        // SAFETY: live function handle; the attribute is a scheduling hint.
        check("cuFuncSetAttribute", unsafe { (d.fns.cuFuncSetAttribute)(self.raw, CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT, percent) })
    }

    /// Launches the kernel asynchronously on `stream`.
    ///
    /// # Safety
    /// `params` must point to values whose types, order and count exactly match the
    /// kernel's C signature, and every device pointer passed must be valid for the
    /// accesses the kernel performs. Typed wrappers in `astral-kernels` uphold this.
    pub unsafe fn launch(&self, cfg: LaunchConfig, stream: &Stream, params: &mut [*mut c_void]) -> Result<()> {
        self.module.ctx.bind()?;
        let d = drv()?;
        if self.module.ctx.pdl() {
            let mut attr = sys::CUlaunchAttribute { id: sys::CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, pad: [0; 4], value: [0; 64] };
            attr.value[0] = 1; // programmaticStreamSerializationAllowed
            let lc = sys::CUlaunchConfig {
                grid_x: cfg.grid.0,
                grid_y: cfg.grid.1,
                grid_z: cfg.grid.2,
                block_x: cfg.block.0,
                block_y: cfg.block.1,
                block_z: cfg.block.2,
                shared_mem_bytes: cfg.shared_mem_bytes,
                stream: stream.raw,
                attrs: &mut attr,
                num_attrs: 1,
            };
            // SAFETY: the caller guarantees `params` matches the kernel signature (see above).
            return check("cuLaunchKernelEx", unsafe { (d.fns.cuLaunchKernelEx)(&lc, self.raw, params.as_mut_ptr(), std::ptr::null_mut()) });
        }
        // SAFETY: as above.
        check("cuLaunchKernel", unsafe {
            (d.fns.cuLaunchKernel)(self.raw, cfg.grid.0, cfg.grid.1, cfg.grid.2, cfg.block.0, cfg.block.1, cfg.block.2, cfg.shared_mem_bytes, stream.raw, params.as_mut_ptr(), std::ptr::null_mut())
        })
    }
}

/// Converts a reference to a kernel-parameter pointer for [`Function::launch`].
pub fn param<T>(v: &mut T) -> *mut c_void {
    v as *mut T as *mut c_void
}
