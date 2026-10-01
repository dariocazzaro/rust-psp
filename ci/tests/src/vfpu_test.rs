use core::f32::consts::PI;

use psp::test_runner::TestRunner;

#[repr(C, align(16))]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Vec4 {
    x: f32,
    y: f32,
    z: f32,
    w: f32,
}

fn vec4(x: f32, y: f32, z: f32, w: f32) -> Vec4 {
    Vec4 { x, y, z, w }
}

fn vec4_splat(num: f32) -> Vec4 {
    vec4(num, num, num, num)
}

macro_rules! vfpu_asm_return {
    ($($l:literal),*,$($i:ident$(.$e:expr)?),*,) => {
        unsafe {
            let mut o = core::mem::MaybeUninit::uninit();
            psp::vfpu_asm!(
                $($l,)*
                in(reg) (&mut o),
                $(in(reg) (&$i$(.$e)*),)*
                options(nostack),
            );
            o.assume_init()
        }
    };
}

pub fn test_main(test_runner: &mut TestRunner) {
    test_runner.check_list(&[
        ("vfpu_add_s_0", test_add_s(0.0), 0.0),
        ("vfpu_add_s_1", test_add_s(1.0), 1.0),
        ("vfpu_sub_s_0", test_sub_s(0.0), 0.0),
        ("vfpu_sub_s_1", test_sub_s(1.0), -1.0),
        ("vfpu_mul_s_0", test_mul_s(0.0), 0.0),
        ("vfpu_mul_s_2", test_mul_s(2.0), 2.0),
        ("vfpu_div_s_1", test_div_s(1.0), 1.0),
        ("vfpu_div_s_2", test_div_s(2.0), 0.5),
        ("vfpu_dot_1", test_dot(1.0), 4.0),
        ("vfpu_dot_2", test_dot(2.0), 8.0),
    ]);
    test_runner.check_list(&[
        ("vfpu_add_q_0", test_add_q(0.0), vec4_splat(0.0)),
        ("vfpu_add_q_1", test_add_q(1.0), vec4_splat(1.0)),
        ("vfpu_sub_q_0", test_sub_q(0.0), vec4_splat(0.0)),
        ("vfpu_sub_q_1", test_sub_q(1.0), vec4_splat(-1.0)),
        ("vfpu_mul_q_0", test_mul_q(0.0), vec4_splat(0.0)),
        ("vfpu_mul_q_2", test_mul_q(2.0), vec4_splat(2.0)),
        ("vfpu_div_q_1", test_div_q(1.0), vec4_splat(1.0)),
        ("vfpu_div_q_2", test_div_q(2.0), vec4_splat(0.5)),
        (
            "vfpu_test_add_vcst_pi_0",
            test_add_vcst_pi(0.0),
            vec4_splat(PI),
        ),
        (
            "vfpu_test_add_vcst_pi_1",
            test_add_vcst_pi(1.0),
            vec4_splat(1.0 + PI),
        ),
        (
            "vfpu_test_shuffle_add_rev",
            test_shuffle_add_rev(vec4(1.0, 2.0, 3.0, 4.0)),
            vec4(4.0, 3.0, 2.0, 1.0),
        ),
        (
            "vfpu_test_shuffle_sub_rev",
            test_shuffle_sub_rev(vec4(1.0, 2.0, 3.0, 4.0)),
            vec4(-4.0, -3.0, -2.0, -1.0),
        ),
    ]);
    test_runner.check(
        "vfpu_mtv_mfv_named_gpr",
        test_mtv_mfv_named_gpr(0x1234_5678),
        0x1234_5678,
    );
}

fn test_add_s(value: f32) -> f32 {
    let zero: f32 = 0.0;
    vfpu_asm_return!(
        "lv.s S010, {1}",
        "lv.s S020, {2}",
        "vadd.s S000, S010, S020",
        "sv.s S000, {0}",
        zero,
        value,
    )
}

fn test_sub_s(value: f32) -> f32 {
    let zero: f32 = 0.0;
    vfpu_asm_return!(
        "lv.s S010, {1}",
        "lv.s S020, {2}",
        "vsub.s S000, S010, S020",
        "sv.s S000, {0}",
        zero,
        value,
    )
}

fn test_mul_s(value: f32) -> f32 {
    let one: f32 = 1.0;
    vfpu_asm_return!(
        "lv.s S010, {1}",
        "lv.s S020, {2}",
        "vmul.s S000, S010, S020",
        "sv.s S000, {0}",
        one,
        value,
    )
}

fn test_div_s(value: f32) -> f32 {
    let one: f32 = 1.0;
    vfpu_asm_return!(
        "lv.s S010, {1}",
        "lv.s S020, {2}",
        "vdiv.s S000, S010, S020",
        "sv.s S000, {0}",
        one,
        value,
    )
}

fn test_dot(num: f32) -> f32 {
    let one = vec4_splat(1.0);
    let value = vec4_splat(num);
    vfpu_asm_return!(
        "lv.q C010, {1}",
        "lv.q C020, {2}",
        "vdot.q S000, C010, C020",
        "sv.s S000, {0}",
        one,
        value,
    )
}

fn test_add_q(num: f32) -> Vec4 {
    let zero = vec4_splat(0.0);
    let value = vec4_splat(num);
    vfpu_asm_return!(
        "lv.q C010, {1}",
        "lv.q C020, {2}",
        "vadd.q C000, C010, C020",
        "sv.q C000, {0}",
        zero,
        value,
    )
}

fn test_sub_q(num: f32) -> Vec4 {
    let zero = vec4_splat(0.0);
    let value = vec4_splat(num);
    vfpu_asm_return!(
        "lv.q C010, {1}",
        "lv.q C020, {2}",
        "vsub.q C000, C010, C020",
        "sv.q C000, {0}",
        zero,
        value,
    )
}

fn test_mul_q(num: f32) -> Vec4 {
    let one = vec4_splat(1.0);
    let value = vec4_splat(num);
    vfpu_asm_return!(
        "lv.q C010, {1}",
        "lv.q C020, {2}",
        "vmul.q C000, C010, C020",
        "sv.q C000, {0}",
        one,
        value,
    )
}

fn test_div_q(num: f32) -> Vec4 {
    let one = vec4_splat(1.0);
    let value = vec4_splat(num);
    vfpu_asm_return!(
        "lv.q C010, {1}",
        "lv.q C020, {2}",
        "vdiv.q C000, C010, C020",
        "sv.q C000, {0}",
        one,
        value,
    )
}

fn test_add_vcst_pi(num: f32) -> Vec4 {
    let value = vec4_splat(num);
    vfpu_asm_return!(
        "lv.q C010, {1}",
        "vcst.q C020, VFPU_PI",
        "vadd.q C000, C010, C020",
        "sv.q C000, {0}",
        value,
    )
}

fn test_shuffle_add_rev(value: Vec4) -> Vec4 {
    let zero = vec4_splat(0.0);
    vfpu_asm_return!(
        "lv.q C010, {1}",
        "lv.q C020, {2}",
        "vadd.q C000, C010, C020[W,Z,Y,X]",
        "sv.q C000, {0}",
        zero,
        value,
    )
}

fn test_shuffle_sub_rev(value: Vec4) -> Vec4 {
    let zero = vec4_splat(0.0);
    vfpu_asm_return!(
        "lv.q C010, {1}",
        "lv.q C020, {2}",
        "vadd.q C000, C010, C020[-W,-Z,-Y,-X]",
        "sv.q C000, {0}",
        zero,
        value,
    )
}

/// Regression test for #199: round-trips a value through S000 with `mtv` and
/// `mfv` while `$2`-`$25` are clobbered. That leaves too few registers that
/// LLVM prints by number for both operands, so at least one of them gets a
/// register it prints by name (`$gp`, `$fp`, ...).
fn test_mtv_mfv_named_gpr(value: u32) -> u32 {
    let out: u32;
    // SAFETY: the asm touches no memory and only writes `out`, the declared
    // clobbers and S000. The compiler never allocates VFPU registers, and like
    // the other tests here this one relies on nothing being live in S000.
    unsafe {
        psp::vfpu_asm!(
            "mtv {value}, S000",
            "mfv {out}, S000",
            value = in(reg) value,
            out = out(reg) out,
            // `out` rather than `lateout`, so `value` can't use these either.
            out("$2") _, out("$3") _, out("$4") _, out("$5") _,
            out("$6") _, out("$7") _, out("$8") _, out("$9") _,
            out("$10") _, out("$11") _, out("$12") _, out("$13") _,
            out("$14") _, out("$15") _, out("$16") _, out("$17") _,
            out("$18") _, out("$19") _, out("$20") _, out("$21") _,
            out("$22") _, out("$23") _, out("$24") _, out("$25") _,
            options(nostack, nomem),
        );
    }
    out
}
