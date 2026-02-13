use any_uart::{Receiver, Sender};
use axplat::console::ConsoleIf;
use fdt_parser::Fdt;
use somehal::boot_info;
use spin::Mutex;

static TX: Mutex<Option<Sender>> = Mutex::new(None);
static RX: Mutex<Option<Receiver>> = Mutex::new(None);

pub(crate) fn setup_early() -> Option<()> {
    let ptr = boot_info().fdt?;
    let fdt = Fdt::from_ptr(ptr).ok()?;
    let choson = fdt.chosen()?;
    let node = choson.debugcon()?;

    let mut uart = any_uart::Uart::new_by_fdt_node(&node, somehal::mem::phys_to_virt)?;
    *TX.lock() = uart.tx.take();
    *RX.lock() = uart.rx.take();

    Some(())
}

struct ConsoleIfImpl;

#[impl_plat_interface]
impl ConsoleIf for ConsoleIfImpl {
    /// Writes given bytes to the console.
    fn write_bytes(bytes: &[u8]) {
        let mut g = TX.lock();
        if let Some(tx) = g.as_mut() {
            macro_rules! write_byte {
                ($b:expr) => {
                    let _ = any_uart::block!(tx.write($b));
                };
            }

            for &c in bytes {
                match c {
                    b'\n' => {
                        write_byte!(b'\r');
                        write_byte!(b'\n');
                    }
                    c => {
                        write_byte!(c);
                    }
                }
            }
        }
    }

    /// Reads bytes from the console into the given mutable slice.
    ///
    /// Returns the number of bytes read.
    fn read_bytes(bytes: &mut [u8]) -> usize {
        let mut read_len = 0;
        while read_len < bytes.len() {
            if let Some(c) = getchar() {
                bytes[read_len] = c;
            } else {
                break;
            }
            read_len += 1;
        }
        read_len
    }
}

fn getchar() -> Option<u8> {
    let mut g = RX.lock();
    if let Some(rx) = g.as_mut() {
        rx.read().ok()
    } else {
        None
    }
}
