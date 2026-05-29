"""BTHACK RDRAM signature validation for the EmuLoader client.

Kept import-light (no CommonClient / emu_loader imports) so it can be referenced from the world's
``__init__.py`` as ``n64_validation_function`` without pulling client-only modules in at world
definition / generation time.

``validate_bt_signature(pm, rdram_base)`` matches EmuLoader's ``ValidationFunc`` contract: it is
called during RDRAM-base detection with a ``ProcessMemory``-like object (anything exposing
``read_bytes(address, length) -> bytes``) and the candidate base address.
"""

from __future__ import annotations

from typing import Any

RDRAM_BASE = 0x80000000  # KSEG0 start; RDRAM mirror
RDRAM_SIZE = 0x800000  # 8 MB with expansion pak (required by BT)
BTHACK_ANCHOR_OFFSET = 0x400000  # physical RDRAM offset of AP_MEMORY_PTR
BTHACK_STRUCT_SIZE = 52
BTHACK_SUB_POINTER_OFFSETS = (
    0x04,  # pc
    0x08,  # pc_message
    0x0C,  # signpost_messages
    0x10,  # pc_settings
    0x14,  # pc_items
    0x18,  # pc_traps
    0x1C,  # pc_exit_map
    0x20,  # n64
    0x24,  # n64_saves_real
    0x28,  # n64_saves_fake
    0x2C,  # n64_saves_nests
    0x30,  # n64_saves_signposts
)


def is_rdram_pointer(value: int) -> bool:
    return RDRAM_BASE <= value < RDRAM_BASE + RDRAM_SIZE


def validate_bt_signature(pm: Any, rdram_base: int) -> bool:
    """Return True if ``rdram_base`` looks like AP-Banjo-Tooie RDRAM.

    - u32 at ``rdram_base + 0x400000`` must be a valid 0x80xxxxxx pointer (BTHACK's
      ``AP_MEMORY_PTR``).
    - At the dereferenced ``ap_memory_ptr_t`` struct, all 12 sub-pointers at offsets 0x04..0x30
      must themselves be valid RDRAM pointers. The patch's ``inject_hooks()`` populates every one
      of them at game boot.
    """
    try:
        anchor = int.from_bytes(pm.read_bytes(rdram_base + BTHACK_ANCHOR_OFFSET, 4), "little")
    except Exception:
        return False
    if not is_rdram_pointer(anchor):
        return False
    physical = anchor & 0x7FFFFFFF
    if physical + BTHACK_STRUCT_SIZE > RDRAM_SIZE:
        return False
    try:
        struct_bytes = pm.read_bytes(rdram_base + physical, BTHACK_STRUCT_SIZE)
    except Exception:
        return False
    for offset in BTHACK_SUB_POINTER_OFFSETS:
        sub_ptr = int.from_bytes(struct_bytes[offset:offset + 4], "little")
        if not is_rdram_pointer(sub_ptr):
            return False
    return True
