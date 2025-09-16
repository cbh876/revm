use crate::precompile_provider::PrecompileProvider;
use context_interface::{journaled_state::JournalTr, ContextTr, local::LocalContextTr};
use interpreter::{CallInput, CallInputs, Gas, InstructionResult, InterpreterResult};
use primitives::{B256, Bytes};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use libloading::Library;

/// 包装器：在命中指定合约 code hash 时，改为调用 AOT 动态库返回结果；否则退回内置 precompiles。
#[derive(Debug, Clone)]
pub struct AotPrecompiles<P> {
    inner: P,
    target_hash: B256,
    so_path: PathBuf,
    lib: OnceLock<Arc<Library>>, 
}

impl<P> AotPrecompiles<P> {
    pub fn new(inner: P, target_hash: B256, so_path: impl Into<PathBuf>) -> Self {
        Self { inner, target_hash, so_path: so_path.into(), lib: OnceLock::new() }
    }
}

impl<CTX, P> PrecompileProvider<CTX> for AotPrecompiles<P>
where
    CTX: ContextTr,
    P: PrecompileProvider<CTX, Output = InterpreterResult>,
{
    type Output = InterpreterResult;

    fn set_spec(&mut self, spec: <CTX::Cfg as context::Cfg>::Spec) -> bool {
        self.inner.set_spec(spec)
    }

    fn run(&mut self, ctx: &mut CTX, inputs: &CallInputs) -> Result<Option<Self::Output>, String> {
        // 先尝试命中 code_hash，再回退到内置 precompiles
        if let Ok(acc) = ctx.journal_mut().load_account_code(inputs.bytecode_address) {
            let code_hash = acc.info.code_hash();
            if code_hash == self.target_hash {
               // eprintln!("LALALAAL AOT hit: addr={:?}", inputs.bytecode_address);
                // 调用 AOT 动态库：contract_entry(in_ptr, in_len, out_ptr, out_cap) -> i32
                unsafe {
                    let lib = if let Some(l) = self.lib.get() {
                        l
                    } else {
                        // try initialize without OnceLock::get_or_try_init (stable)
                        let l = Library::new(&self.so_path)
                            .map(Arc::new)
                            .map_err(|e| format!("load so failed: {e}"))?;
                        let _ = self.lib.set(l);
                        self.lib.get().expect("lib set")
                    };
                    let entry: libloading::Symbol<'_, unsafe extern "C" fn(*const u8, usize, *mut u8, usize) -> i32> =
                        (**lib).get(b"custom").map_err(|e| format!("dlsym custom failed: {e}"))?;

                    // 读取输入并对齐到 32 字节大端（与 Solidity ABI 单参数一致）
                    let mut input_padded: [u8; 32] = [0u8; 32];
                    let input_bytes: &[u8];
                    let r;
                    match &inputs.input {
                        CallInput::SharedBuffer(range) => {
                            if let Some(slice) = ctx
                                .local()
                                .shared_memory_buffer_slice(range.clone())
                            {
                                r = slice;
                                input_bytes = r.as_ref();
                            } else {
                                input_bytes = &[];
                            }
                        }
                        CallInput::Bytes(bytes) => {
                            input_bytes = bytes.as_ref();
                        }
                    }
                    if input_bytes.len() <= 32 {
                        let start = 32 - input_bytes.len();
                        input_padded[start..].copy_from_slice(input_bytes);
                    } else {
                        // 超过 32 字节则截断保留低位（大端输入）
                        input_padded.copy_from_slice(&input_bytes[input_bytes.len() - 32..]);
                    }

                    // 输出缓冲：默认按 32 字节（U256）分配，避免冗长 0 填充
                    let mut out = vec![0u8; 32];
                    let status = entry(
                        input_padded.as_ptr(),
                        input_padded.len(),
                        out.as_mut_ptr(),
                        out.len(),
                    );
                    // 兼容两种语义：
                    // 1) 返回 0/非0 表示 成功/失败（原语义）
                    // 2) 返回写入的字节数（>=0 表示成功，值为写入长度）
                    let status_isize = status as isize;
                    let is_ok = status_isize >= 0;
                    if status_isize > 0 {
                        let written_len = (status_isize as usize).min(out.len());
                        out.truncate(written_len);
                    }
                    let result = InterpreterResult {
                        result: if is_ok { InstructionResult::Return } else { InstructionResult::Revert },
                        gas: Gas::new(inputs.gas_limit),
                        output: Bytes::from(out),
                    };
                    return Ok(Some(result));
                }
            }
        }

        // 未命中：走默认 precompiles
        self.inner.run(ctx, inputs)
    }

    fn warm_addresses(&self) -> Box<impl Iterator<Item = primitives::Address>> {
        self.inner.warm_addresses()
    }

    fn contains(&self, address: &primitives::Address) -> bool {
        self.inner.contains(address)
    }
}

