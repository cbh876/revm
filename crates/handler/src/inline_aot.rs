//! Inline AOT wrapper: allows intercepting frame_run with a user callback that
//! has access to Interpreter and Context, enabling custom execution paths.

use crate::{
    evm::{EvmTr, FrameTr},
    instructions::InstructionProvider,
    item_or_result::FrameInitOrResult,
    FrameResult, ItemOrResult, PrecompileProvider,
};
use primitives::{Address, B256, U256, Bytes, keccak256};
use crate::api::{ExecuteEvm, ExecuteCommitEvm};
use crate::{MainnetHandler, Handler};
use context::{ContextSetters};
use context_interface::{JournalTr};
use state::EvmState;
use context::result::{ExecutionResult, HaltReason, EVMError, ResultAndState, InvalidTransaction};
use context::Database;
use context::{ContextTr, Evm as BaseEvm, FrameStack};
use interpreter::{interpreter::EthInterpreter, InterpreterAction, InterpreterResult, InterpreterTypes};
use revm_ffi_bridge::{FfiCallData, FfiEntryFn, FfiReturnData, FfiHostVTable, TRANSFER_SELECTOR};
use libloading::Library;
use std::{path::PathBuf, sync::{Arc, OnceLock}};
use interpreter::interpreter_types::InputsTr;
use database_interface::DatabaseCommit;


/// EVM wrapper that intercepts `frame_run` and invokes a user callback.
///
/// The callback receives `&mut Interpreter` and `&mut Context` and can return
/// a custom `InterpreterAction` (e.g. by running an AOT path). Returning `None`
/// falls back to the normal interpreter loop.
pub struct InlineAotEvm<CTX, INSP, I, P, F>
where
    CTX: ContextTr,
    I: InstructionProvider<Context = CTX, InterpreterTypes = EthInterpreter>,
    P: PrecompileProvider<CTX, Output = InterpreterResult>,
    F: FrameTr,
{
    /// Inner EVM.
    pub inner: BaseEvm<CTX, INSP, I, P, F>,
    /// Callback invoked before falling back to the normal interpreter run.
    pub callback: Box<dyn FnMut(&mut interpreter::Interpreter<EthInterpreter>, &mut CTX) -> Option<InterpreterAction>>,
    /// Optional dl callback (FFI) for invoking external AOT entry.
    pub dl: Option<InlineDl>,
    /// Whether current frame has already been handled by FFI.
    handled_current_frame: bool,
    /// Whether current transaction has already been handled by FFI (prevent re-entry this tx).
    handled_this_tx: bool,
}

impl<CTX, INSP, I, P, F> InlineAotEvm<CTX, INSP, I, P, F>
where
    CTX: ContextTr,
    I: InstructionProvider<Context = CTX, InterpreterTypes = EthInterpreter>,
    P: PrecompileProvider<CTX, Output = InterpreterResult>,
    F: FrameTr,
{
    /// Wrap an existing EVM with the given callback.
    pub fn from_evm(
        inner: BaseEvm<CTX, INSP, I, P, F>,
        callback: impl FnMut(&mut interpreter::Interpreter<EthInterpreter>, &mut CTX) -> Option<InterpreterAction> + 'static,
    ) -> Self {
        Self { inner, callback: Box::new(callback), dl: None, handled_current_frame: false, handled_this_tx: false }
    }

    /// Enable FFI dl callback with a given library path and symbol name.
    pub fn with_ffi(mut self, lib_path: PathBuf, symbol: &'static [u8]) -> Self {
        self.dl = Some(InlineDl::new(lib_path, symbol));
        self
    }
}

impl<CTX, INSP, I, P> EvmTr
    for InlineAotEvm<CTX, INSP, I, P, crate::EthFrame<EthInterpreter>>
where
    CTX: ContextTr,
    I: InstructionProvider<Context = CTX, InterpreterTypes = EthInterpreter>,
    P: PrecompileProvider<CTX, Output = InterpreterResult>,
{
    type Context = CTX;
    type Instructions = I;
    type Precompiles = P;
    type Frame = crate::EthFrame<EthInterpreter>;

    #[inline]
    fn ctx(&mut self) -> &mut Self::Context { &mut self.inner.ctx }
    #[inline]
    fn ctx_ref(&self) -> &Self::Context { &self.inner.ctx }
    #[inline]
    fn frame_stack(&mut self) -> &mut FrameStack<Self::Frame> { &mut self.inner.frame_stack }

    #[inline]
    fn frame_init(
        &mut self,
        frame_input: <Self::Frame as FrameTr>::FrameInit,
    ) -> Result<crate::evm::FrameInitResult<'_, Self::Frame>, crate::evm::ContextDbError<Self::Context>> {
        // reset handled flag for new frame
        self.handled_current_frame = false;
        EvmTr::frame_init(&mut self.inner, frame_input)
    }

    #[inline]
    fn frame_run(
        &mut self,
    ) -> Result<crate::item_or_result::FrameInitOrResult<Self::Frame>, crate::evm::ContextDbError<Self::Context>> {
        // Intercept current frame and try AOT callback
        let frame = self.inner.frame_stack.get();
        let ctx = &mut self.inner.ctx;

        if let Some(action) = (self.callback)(&mut frame.interpreter, ctx) {
            // process the custom action
            let res = frame.process_next_action(ctx, action);
            if let Ok(ref r) = res {
                if r.is_result() {
                    frame.set_finished(true);
                }
            }
            return res;
        }
        // FFI callback path
        if let Some(dl) = &self.dl {
            if !self.handled_current_frame && !self.handled_this_tx {
                // Only allow FFI to short-circuit at top-level frame (depth==0)
                if frame.depth == 0 {
                    if let Some(action) = dl.try_call(&mut frame.interpreter, ctx) {
                        let target = frame.interpreter.input.target_address();
                        let caller = ctx.caller();
                        let _ = ctx.journal_mut().load_account(target);
                        let _ = ctx.journal_mut().load_account(caller);
                        // mark handled so we don't re-enter FFI for the same frame
                        self.handled_current_frame = true;
                        self.handled_this_tx = true;
                        // Disable FFI for the rest of this transaction
                        let res = frame.process_next_action(ctx, action);
                        if let Ok(ref r) = res {
                            if r.is_result() {
                                frame.set_finished(true);
                            }
                        }
                        return res;
                    }
                }
            }
        }
        
        // Fallback to normal interpreter run
        let instructions = &mut self.inner.instruction;
        let action = frame
            .interpreter
            .run_plain(instructions.instruction_table(), ctx);
        // If interpreter returned, also mark handled to avoid repeated FFI when coming back
        if matches!(action, InterpreterAction::Return(..)) {
            self.handled_current_frame = true;
        }
        frame.process_next_action(ctx, action)
    }

    #[inline]
    fn frame_return_result(
        &mut self,
        result: <Self::Frame as FrameTr>::FrameResult,
    ) -> Result<Option<<Self::Frame as FrameTr>::FrameResult>, crate::evm::ContextDbError<Self::Context>> {
        EvmTr::frame_return_result(&mut self.inner, result)
    }

    #[inline]
    fn ctx_instructions(&mut self) -> (&mut Self::Context, &mut Self::Instructions) {
        (&mut self.inner.ctx, &mut self.inner.instruction)
    }

    #[inline]
    fn ctx_precompiles(&mut self) -> (&mut Self::Context, &mut Self::Precompiles) {
        (&mut self.inner.ctx, &mut self.inner.precompiles)
    }
}

/// Simple dl wrapper for calling FFI entry.
pub struct InlineDl {
    lib_path: PathBuf,
    symbol: &'static [u8],
    lib: OnceLock<Arc<Library>>,
}

impl InlineDl {
    pub fn new(lib_path: PathBuf, symbol: &'static [u8]) -> Self {
        Self { lib_path, symbol, lib: OnceLock::new() }
    }

    fn get_or_load(&self) -> Result<&Arc<Library>, String> {
        if let Some(l) = self.lib.get() { return Ok(l); }
        let l = unsafe { Library::new(&self.lib_path).map(Arc::new).map_err(|e| format!("dlopen failed: {e}"))? };
        let _ = self.lib.set(l);
        Ok(self.lib.get().expect("lib set"))
    }

    /// Attempt to call FFI and build an InterpreterAction::Return.
    pub fn try_call<CTX: ContextTr>(&self, interp: &mut interpreter::Interpreter<EthInterpreter>, ctx: &mut CTX) -> Option<InterpreterAction> {
        // Hoist immutable reads before mutable borrows
        let caller = ctx.caller();
        let addr_to_load = interp
        .input
        .bytecode_address()
        .copied()
        .unwrap_or(interp.input.target_address());
        let _ = ctx.journal_mut().load_account(addr_to_load);
        let _ = ctx.journal_mut().load_account(caller);
        // Build calldata from interpreter input
        let input = match &interp.input.input {
            interpreter::interpreter_action::CallInput::Bytes(b) => b.as_ref(),
            interpreter::interpreter_action::CallInput::SharedBuffer(_) => {
                // Not handling shared buffer in this minimal path
                return None;
            }
        };
        // 仅在识别到特定 selector 时走 FFI
        if input.len() < 4 || &input[..4] != &TRANSFER_SELECTOR { return None; }
        let lib = self.get_or_load().ok()?;
        unsafe {
            let entry: libloading::Symbol<'_, FfiEntryFn> = (**lib).get(self.symbol).ok()?;
            // Host shims
            // Thread-local target address captured before FFI call
            thread_local! {
                static TL_TARGET: core::cell::Cell<Address> = core::cell::Cell::new(Address::ZERO);
            }
            extern "C" fn sload_shim<CTX: ContextTr>(host_ctx: *mut core::ffi::c_void, addr_ptr: *const u8, key_ptr: *const u8, out_ptr: *mut u8) -> i32 {
                // unsafe {
                //     let ctx = &mut *(host_ctx as *mut CTX);
                //     // Guest provides the storage account address (20 bytes) and storage key (32 bytes)
                //     let mut addr = Address::from_slice(core::slice::from_raw_parts(addr_ptr, 20));
                //     let key  = B256::from_slice(core::slice::from_raw_parts(key_ptr, 32));
                //     // Ensure account is present then perform sload
                //     let _ = ctx.journal_mut().load_account(addr);
                //     addr = hex::decode("0x0000000000000000000000000000000000009999").unwrap().into();
                //     println!("LALALAAL addr: {:?}, key: {:?}", addr, key);
                //     let val = ctx
                //         .sload(addr, key.into())
                //         .map(|l| l.data)
                //         .unwrap_or(U256::ZERO);
                //     println!("LALALAAL sload: {:?}", val);
                //     let be = val.to_be_bytes_vec();
                //     core::ptr::copy_nonoverlapping(be.as_ptr(), out_ptr, 32);
                //     0
                // }
                unsafe {
                    let ctx = &mut *(host_ctx as *mut CTX);
                    // 仍可读取 guest_addr 仅用于调试
                    let _guest_addr = Address::from_slice(core::slice::from_raw_parts(addr_ptr, 20));
                    // 强制使用合约地址作为存储账户
                    let storage_account = {
                        let mut a = Address::ZERO;
                        TL_TARGET.with(|c| { a = c.get(); });
                        a
                    };
                    let key  = B256::from_slice(core::slice::from_raw_parts(key_ptr, 32));
                    let _ = ctx.journal_mut().load_account(storage_account);
                    let val = ctx.sload(storage_account, key.into()).map(|l| l.data).unwrap_or(U256::ZERO);
                    core::ptr::copy_nonoverlapping(val.to_be_bytes_vec().as_ptr(), out_ptr, 32);
                    0
                }
            }
            extern "C" fn sstore_shim<CTX: ContextTr>(host_ctx: *mut core::ffi::c_void, addr_ptr: *const u8, key_ptr: *const u8, val_ptr: *const u8) -> i32 {
                unsafe {
                    // let ctx = &mut *(host_ctx as *mut CTX);
                    // // Guest provides the storage account address (20 bytes) and storage key (32 bytes)
                    // let addr = Address::from_slice(core::slice::from_raw_parts(addr_ptr, 20));
                    // let key  = B256::from_slice(core::slice::from_raw_parts(key_ptr, 32));
                    // let val  = U256::from_be_slice(core::slice::from_raw_parts(val_ptr, 32));
                    // // Ensure account is present then perform sstore
                    // let _ = ctx.journal_mut().load_account(addr);
                    // ctx.sstore(addr, key.into(), val);
                    // 0
                    unsafe {
                        let ctx = &mut *(host_ctx as *mut CTX);
                        // 强制使用合约地址
                        let storage_account = {
                            let mut a = Address::ZERO;
                            TL_TARGET.with(|c| { a = c.get(); });
                            a
                        };
                        let key = B256::from_slice(core::slice::from_raw_parts(key_ptr, 32));
                        let val = U256::from_be_slice(core::slice::from_raw_parts(val_ptr, 32));
                        let _ = ctx.journal_mut().load_account(storage_account);
                        ctx.sstore(storage_account, key.into(), val);
                        0
                    }
                }
            }
            extern "C" fn get_caller_shim<CTX: ContextTr>(host_ctx: *mut core::ffi::c_void, out_addr20: *mut u8) -> i32 {
                unsafe {
                    let ctx = &mut *(host_ctx as *mut CTX);
                    let addr = ctx.caller();
                    core::ptr::copy_nonoverlapping(addr.as_ptr(), out_addr20, 20);
                    0
                }
            }

            let vtable = FfiHostVTable {
                sload: sload_shim::<CTX>,
                sstore: sstore_shim::<CTX>,
                get_caller: get_caller_shim::<CTX>,
            };
            // Set current target for shims
            TL_TARGET.with(|c| { c.set(interp.input.target_address()); });
            // Ensure the contract account is loaded in journal to satisfy subsequent sload calls
            {
                let _ = ctx.journal_mut().load_account(addr_to_load);
            }
            let mut out = [0u8; 64]; // enough for 2 words

            let written = entry(
                ctx as *mut _ as *mut core::ffi::c_void,
                FfiCallData { ptr: input.as_ptr(), len: input.len() },
                FfiReturnData { ptr: out.as_mut_ptr(), cap: out.len() },
                &vtable as *const FfiHostVTable,
            );
            if written < 0 { return None; }
            let len = core::cmp::min(written as usize, 32);

            // 清空本帧输入，避免下一轮 selector 继续命中 FFI
            interp.input.input = interpreter::interpreter_action::CallInput::Bytes(Bytes::new());
            let bytes = Bytes::copy_from_slice(&out[..len]);
            Some(InterpreterAction::new_return(interpreter::InstructionResult::Return, bytes, interp.gas))
        }
    }
}

// Implement ExecuteEvm for InlineAotEvm so callers can use evm.transact(...)
impl<CTX, INSP, I, P> ExecuteEvm for InlineAotEvm<CTX, INSP, I, P, crate::EthFrame<EthInterpreter>>
where
    CTX: ContextTr<Journal: JournalTr<State = EvmState>> + ContextSetters,
    I: InstructionProvider<Context = CTX, InterpreterTypes = EthInterpreter>,
    P: PrecompileProvider<CTX, Output = InterpreterResult>,
{
    type ExecutionResult = ExecutionResult<HaltReason>;
    type State = EvmState;
    type Error = EVMError<<CTX::Db as Database>::Error, InvalidTransaction>;
    type Tx = <CTX as ContextTr>::Tx;
    type Block = <CTX as ContextTr>::Block;

    #[inline]
    fn set_block(&mut self, block: Self::Block) {
        self.inner.ctx.set_block(block);
    }

    #[inline]
    fn transact_one(&mut self, tx: Self::Tx) -> Result<Self::ExecutionResult, Self::Error> {
        self.inner.ctx.set_tx(tx);
        MainnetHandler::default().run(self)
    }

    #[inline]
    fn finalize(&mut self) -> Self::State {
        // reset transaction-level FFI handling flag
        self.handled_current_frame = false;
        self.handled_this_tx = false;
        self.ctx().journal_mut().finalize()
    }

    #[inline]
    fn replay(&mut self) -> Result<ResultAndState<HaltReason>, Self::Error> {
        MainnetHandler::default().run(self).map(|result| {
            let state = self.finalize();
            ResultAndState::new(result, state)
        })
    }
}

// Implement ExecuteCommitEvm so callers can use transact_commit(...)
impl<CTX, INSP, I, P> ExecuteCommitEvm for InlineAotEvm<CTX, INSP, I, P, crate::EthFrame<EthInterpreter>>
where
    CTX: ContextTr<Journal: JournalTr<State = EvmState>, Db: DatabaseCommit> + ContextSetters,
    I: InstructionProvider<Context = CTX, InterpreterTypes = EthInterpreter>,
    P: PrecompileProvider<CTX, Output = InterpreterResult>,
{
    #[inline]
    fn commit(&mut self, state: Self::State) {
        self.inner.commit(state);
    }
}


