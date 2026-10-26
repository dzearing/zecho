use std::sync::atomic::AtomicBool;

/// Whether hold-to-record needs Shift held with Fn. Fn alone collides with the
/// macOS Globe key action (emoji picker, input source, dictation).
static FN_NEEDS_SHIFT: AtomicBool = AtomicBool::new(true);

/// Whether the record key is held after a modifier change. Starting needs Fn (plus
/// Shift if required); once started, only releasing Fn ends it, so letting go of
/// Shift early doesn't cut the recording short.
fn next_fn_active(was_active: bool, fn_down: bool, shift_down: bool, needs_shift: bool) -> bool {
    if was_active {
        fn_down
    } else {
        fn_down && (shift_down || !needs_shift)
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use core_foundation::runloop::{kCFRunLoopCommonModes, CFRunLoop};
    use core_graphics::event::{
        CGEvent, CGEventFlags, CGEventTap, CGEventTapLocation, CGEventTapOptions,
        CGEventTapPlacement, CGEventType,
    };
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use tauri::{AppHandle, Emitter};

    static FN_IS_DOWN: AtomicBool = AtomicBool::new(false);

    pub fn set_fn_needs_shift(needs_shift: bool) {
        super::FN_NEEDS_SHIFT.store(needs_shift, Ordering::SeqCst);
    }

    pub fn start_fn_key_listener(app_handle: AppHandle) {
        std::thread::spawn(move || {
            let tap = CGEventTap::new(
                CGEventTapLocation::Session,
                CGEventTapPlacement::HeadInsertEventTap,
                CGEventTapOptions::ListenOnly,
                vec![CGEventType::FlagsChanged],
                move |_proxy, _event_type, event| {
                    {
                        let flags = event.get_flags();
                        let fn_pressed = super::next_fn_active(
                            FN_IS_DOWN.load(Ordering::SeqCst),
                            flags.contains(CGEventFlags::CGEventFlagSecondaryFn),
                            flags.contains(CGEventFlags::CGEventFlagShift),
                            super::FN_NEEDS_SHIFT.load(Ordering::SeqCst),
                        );
                        let was_down = FN_IS_DOWN.swap(fn_pressed, Ordering::SeqCst);

                        if fn_pressed && !was_down {
                            let _ = app_handle.emit("fn-key-down", ());
                        } else if !fn_pressed && was_down {
                            let _ = app_handle.emit("fn-key-up", ());
                        }
                    }
                    None
                },
            );

            match tap {
                Ok(tap) => {
                    unsafe {
                        let loop_source = tap.mach_port.create_runloop_source(0).unwrap();
                        let run_loop = CFRunLoop::get_current();
                        run_loop.add_source(&loop_source, kCFRunLoopCommonModes);
                        tap.enable();
                        CFRunLoop::run_current();
                    }
                }
                Err(()) => {
                    eprintln!(
                        "Failed to create CGEventTap. Grant Accessibility permissions: \
                         System Settings > Privacy & Security > Accessibility"
                    );
                }
            }
        });
    }
}

#[cfg(target_os = "macos")]
pub use macos::{set_fn_needs_shift, start_fn_key_listener};

#[cfg(not(target_os = "macos"))]
pub fn set_fn_needs_shift(needs_shift: bool) {
    FN_NEEDS_SHIFT.store(needs_shift, std::sync::atomic::Ordering::SeqCst);
}

#[cfg(not(target_os = "macos"))]
pub fn start_fn_key_listener(_app_handle: tauri::AppHandle) {
    eprintln!("FN key listener is only supported on macOS");
}

#[cfg(test)]
mod tests {
    use super::next_fn_active;

    #[test]
    fn fn_alone_starts_when_shift_not_required() {
        assert!(next_fn_active(false, true, false, false));
    }

    #[test]
    fn fn_alone_is_ignored_when_shift_required() {
        assert!(!next_fn_active(false, true, false, true));
    }

    #[test]
    fn shift_fn_starts_when_shift_required() {
        assert!(next_fn_active(false, true, true, true));
    }

    #[test]
    fn releasing_shift_keeps_recording() {
        assert!(next_fn_active(true, true, false, true));
    }

    #[test]
    fn releasing_fn_stops() {
        assert!(!next_fn_active(true, false, true, true));
    }
}
