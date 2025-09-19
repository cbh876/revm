#![cfg_attr(not(feature = "std"), no_std)]

use core::ffi::c_void;

#[repr(C)]
pub struct FfiCallData {
    pub ptr: *const u8,
    pub len: usize,
}

#[repr(C)]
pub struct FfiReturnData {
    pub ptr: *mut u8,
    pub cap: usize,
}

#[repr(C)]
pub struct FfiHostVTable {
    pub sload: unsafe extern "C" fn(host_ctx: *mut c_void, addr_ptr: *const u8, key_ptr: *const u8, out_ptr: *mut u8) -> i32,
    pub sstore: unsafe extern "C" fn(host_ctx: *mut c_void, addr_ptr: *const u8, key_ptr: *const u8, val_ptr: *const u8) -> i32,
    pub get_caller: unsafe extern "C" fn(host_ctx: *mut c_void, out_addr20: *mut u8) -> i32,
}

/// Unified plugin entry function type.
/// Returns: >=0 bytes written, <0 error.
pub type FfiEntryFn = unsafe extern "C" fn(
    host_ctx: *mut c_void,
    calldata: FfiCallData,
    ret: FfiReturnData,
    host: *const FfiHostVTable,
) -> i32;

pub const TRANSFER_SELECTOR: [u8; 4] = [0xa9, 0x05, 0x9c, 0xbb];

