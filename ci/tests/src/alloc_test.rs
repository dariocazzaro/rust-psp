use alloc::alloc::{alloc, dealloc, Layout};
use psp::test_runner::TestRunner;

const ALIGNS: [usize; 13] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
const SIZES: [usize; 4] = [1, 3, 64, 1000];
const COUNT: usize = ALIGNS.len() * SIZES.len();

fn pattern(block: usize, offset: usize) -> u8 {
    (block as u8).wrapping_mul(37) ^ offset as u8
}

pub fn test_main(test_runner: &mut TestRunner) {
    let mut layouts = [Layout::new::<u8>(); COUNT];
    let mut ptrs = [core::ptr::null_mut::<u8>(); COUNT];
    let mut all_non_null = true;
    let mut all_aligned = true;
    let mut all_intact = true;

    // The test runner allocates, so it must not be called until every block
    // below has been freed again.
    // SAFETY: a plain syscall with no arguments.
    let free_before = unsafe { psp::sys::sceKernelTotalFreeMemSize() };

    // Keep every block alive at the same time, so that the system hands out
    // block heads at many different addresses relative to each alignment.
    for (i, (layout, ptr)) in layouts.iter_mut().zip(ptrs.iter_mut()).enumerate() {
        let align = ALIGNS[i / SIZES.len()];
        let size = SIZES[i % SIZES.len()];
        *layout = Layout::from_size_align(size, align).unwrap();

        // SAFETY: `layout` has a nonzero size.
        let p = unsafe { alloc(*layout) };
        *ptr = p;
        if p.is_null() {
            all_non_null = false;
            continue;
        }
        if p as usize % align != 0 {
            all_aligned = false;
        }
        for offset in 0..size {
            // SAFETY: `p` is valid for `size` bytes.
            unsafe { p.add(offset).write(pattern(i, offset)) };
        }
    }

    // Check the patterns only after all blocks are written, so that any
    // overlap between blocks shows up.
    for (i, (layout, &ptr)) in layouts.iter().zip(ptrs.iter()).enumerate() {
        if ptr.is_null() {
            continue;
        }
        for offset in 0..layout.size() {
            // SAFETY: `ptr` is valid for `layout.size()` bytes.
            if unsafe { ptr.add(offset).read() } != pattern(i, offset) {
                all_intact = false;
            }
        }
    }

    for (layout, &ptr) in layouts.iter().zip(ptrs.iter()) {
        if !ptr.is_null() {
            // SAFETY: `ptr` was allocated above with `layout`.
            unsafe { dealloc(ptr, *layout) };
        }
    }

    // SAFETY: as above.
    let free_after = unsafe { psp::sys::sceKernelTotalFreeMemSize() };

    test_runner.check_true("alloc_aligned_non_null", all_non_null);
    test_runner.check_true("alloc_aligned_is_aligned", all_aligned);
    test_runner.check_true("alloc_aligned_contents_intact", all_intact);
    test_runner.check("alloc_aligned_dealloc_frees_all", free_after, free_before);
}
