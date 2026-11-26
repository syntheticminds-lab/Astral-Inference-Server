// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Synthetic Minds LTD
//! Raw CUDA driver API, loaded from `libcuda` at runtime.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_void};
use std::sync::OnceLock;

pub type CUresult = i32;
pub type CUdevice = i32;
pub type CUdeviceptr = u64;
pub type CUcontext = *mut c_void;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;

pub type CUgraph = *mut c_void;
pub type CUgraphExec = *mut c_void;
pub type CUmemGenericAllocationHandle = u64;

pub const CUDA_SUCCESS: CUresult = 0;

/// `CUlaunchAttribute` (cuda.h): id, padding, 64-byte value union.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct CUlaunchAttribute {
    pub id: u32,
    pub pad: [u8; 4],
    pub value: [u8; 64],
}

/// `CUlaunchConfig` (cuda.h).
#[repr(C)]
pub struct CUlaunchConfig {
    pub grid_x: u32,
    pub grid_y: u32,
    pub grid_z: u32,
    pub block_x: u32,
    pub block_y: u32,
    pub block_z: u32,
    pub shared_mem_bytes: u32,
    pub stream: CUstream,
    pub attrs: *mut CUlaunchAttribute,
    pub num_attrs: u32,
}

/// `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION`.
pub const CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION: u32 = 6;

/// `CUmemLocation` (cuda.h).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUmemLocation {
    pub type_: u32,
    pub id: i32,
}

/// `CUmemAllocationProp::allocFlags` (cuda.h).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CUmemAllocFlags {
    pub compression_type: u8,
    pub gpu_direct_rdma_capable: u8,
    pub usage: u16,
    pub reserved: [u8; 4],
}

/// `CUmemAllocationProp` (cuda.h).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUmemAllocationProp {
    pub type_: u32,
    pub requested_handle_types: u32,
    pub location: CUmemLocation,
    pub win32_handle_meta_data: *mut c_void,
    pub alloc_flags: CUmemAllocFlags,
}

/// `CUmemAccessDesc` (cuda.h).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUmemAccessDesc {
    pub location: CUmemLocation,
    pub flags: u32,
}

pub const CU_MEM_ALLOCATION_TYPE_PINNED: u32 = 1;
pub const CU_MEM_LOCATION_TYPE_DEVICE: u32 = 1;
pub const CU_MEM_ACCESS_FLAGS_PROT_READWRITE: u32 = 3;
pub const CU_MEM_ALLOC_GRANULARITY_RECOMMENDED: u32 = 1;
pub const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: u32 = 1;

api_table! {
    pub struct DriverFns -> CUresult {
        cuInit(u32);
        cuDriverGetVersion(*mut i32);
        cuGetErrorName(CUresult, *mut *const c_char);
        cuDeviceGetCount(*mut i32);
        cuDeviceGet(*mut CUdevice, i32);
        cuDeviceGetName(*mut c_char, i32, CUdevice);
        cuDeviceGetUuid(*mut u8, CUdevice);
        cuDeviceTotalMem_v2(*mut usize, CUdevice);
        cuDeviceGetAttribute(*mut i32, i32, CUdevice);
        cuDevicePrimaryCtxRetain(*mut CUcontext, CUdevice);
        cuDevicePrimaryCtxRelease_v2(CUdevice);
        cuCtxSetCurrent(CUcontext);
        cuCtxSynchronize();
        cuMemGetInfo_v2(*mut usize, *mut usize);
        cuMemAlloc_v2(*mut CUdeviceptr, usize);
        cuMemFree_v2(CUdeviceptr);
        cuMemcpyHtoD_v2(CUdeviceptr, *const c_void, usize);
        cuMemcpyDtoH_v2(*mut c_void, CUdeviceptr, usize);
        cuMemcpyDtoDAsync_v2(CUdeviceptr, CUdeviceptr, usize, CUstream);
        cuMemsetD8_v2(CUdeviceptr, u8, usize);
        cuMemsetD8Async(CUdeviceptr, u8, usize, CUstream);
        cuStreamCreate(*mut CUstream, u32);
        cuStreamDestroy_v2(CUstream);
        cuStreamSynchronize(CUstream);
        cuEventCreate(*mut CUevent, u32);
        cuEventDestroy_v2(CUevent);
        cuEventRecord(CUevent, CUstream);
        cuEventSynchronize(CUevent);
        cuModuleLoadData(*mut CUmodule, *const c_void);
        cuModuleUnload(CUmodule);
        cuModuleGetFunction(*mut CUfunction, CUmodule, *const c_char);
        cuLaunchKernel(CUfunction, u32, u32, u32, u32, u32, u32, u32, CUstream, *mut *mut c_void, *mut *mut c_void);
        cuFuncSetAttribute(CUfunction, i32, i32);
        cuLaunchKernelEx(*const CUlaunchConfig, CUfunction, *mut *mut c_void, *mut *mut c_void);
        cuMemAddressReserve(*mut CUdeviceptr, usize, usize, CUdeviceptr, u64);
        cuMemAddressFree(CUdeviceptr, usize);
        cuMemCreate(*mut CUmemGenericAllocationHandle, usize, *const CUmemAllocationProp, u64);
        cuMemRelease(CUmemGenericAllocationHandle);
        cuMemMap(CUdeviceptr, usize, usize, CUmemGenericAllocationHandle, u64);
        cuMemUnmap(CUdeviceptr, usize);
        cuMemSetAccess(CUdeviceptr, usize, *const CUmemAccessDesc, usize);
        cuMemGetAllocationGranularity(*mut usize, *const CUmemAllocationProp, u32);
        cuStreamBeginCapture_v2(CUstream, u32);
        cuStreamEndCapture(CUstream, *mut CUgraph);
        cuGraphInstantiateWithFlags(*mut CUgraphExec, CUgraph, u64);
        cuGraphLaunch(CUgraphExec, CUstream);
        cuGraphExecDestroy(CUgraphExec);
        cuGraphDestroy(CUgraph);
        cuProfilerStart();
        cuProfilerStop();
        cuTensorMapEncodeTiled(*mut c_void, u32, u32, *mut c_void, *const u64, *const u64, *const u32, *const u32, u32, u32, u32, u32);
    }
}

type ElapsedFn = unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult;

/// The loaded driver. `_lib` keeps the library mapped for the life of the process.
#[allow(non_snake_case)]
pub struct Driver {
    pub fns: DriverFns,
    /// `cuEventElapsedTime_v2` where exported (CUDA 12.8+), else the original symbol.
    pub cuEventElapsedTime: ElapsedFn,
    _lib: libloading::Library,
}

static DRIVER: OnceLock<Result<Driver, String>> = OnceLock::new();

const CANDIDATES: &[&str] = &["libcuda.so.1", "libcuda.so", "nvcuda.dll"];

/// Loads and initialises the driver once per process.
pub fn driver() -> Result<&'static Driver, String> {
    DRIVER
        .get_or_init(|| {
            let mut last = String::from("no candidates tried");
            for name in CANDIDATES {
                // SAFETY: loading the system CUDA driver library; its initialisers are trusted.
                let lib = match unsafe { libloading::Library::new(name) } {
                    Ok(l) => l,
                    Err(e) => {
                        last = format!("{name}: {e}");
                        continue;
                    }
                };
                // SAFETY: symbol signatures follow cuda.h.
                let fns = unsafe { DriverFns::load(&lib)? };
                let elapsed: ElapsedFn = unsafe {
                    match lib.get::<ElapsedFn>(b"cuEventElapsedTime_v2\0") {
                        Ok(s) => *s,
                        Err(_) => *lib.get::<ElapsedFn>(b"cuEventElapsedTime\0").map_err(|e| format!("cuEventElapsedTime: {e}"))?,
                    }
                };
                // SAFETY: cuInit(0) is the documented first call.
                let rc = unsafe { (fns.cuInit)(0) };
                if rc != CUDA_SUCCESS {
                    return Err(format!("cuInit failed with CUresult {rc} (is an NVIDIA GPU present and the driver loaded?)"));
                }
                return Ok(Driver { fns, cuEventElapsedTime: elapsed, _lib: lib });
            }
            Err(format!("CUDA driver library not found ({last})"))
        })
        .as_ref()
        .map_err(|e| e.clone())
}
