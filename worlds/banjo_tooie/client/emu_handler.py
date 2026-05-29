"""Banjo-Tooie handler for the shared EmuLoader Archipelago client.

This replaces the bespoke ``BTClient.py`` context + transport tasks. The world advertises this
class as ``n64_client_handler`` (plus ``n64_validation_function``); the EmuLoader client
discovers it, detects the ROM, and drives :meth:`game_watcher` each tick.

Two transports are supported through one handler:
  * **emulator** (``ctx.transport_mode == "emulator"``): direct RDRAM access via ``ctx`` memory
    methods, wrapped in :class:`_CtxMemoryAdapter` so the existing ``client.state`` /
    ``client.game`` helpers (which expect a ``BTEmuLoaderClient``) work unchanged.
  * **alternate** (EverDrive 64): a socket bridge on ``localhost:21221`` speaking BT's JSON
    payload protocol. Established by :meth:`alternate_connect`; one exchange per tick.

The handler keeps per-session state **on ``ctx``** (see :func:`_init_bt_state`) so the large
existing payload helpers (``get_payload`` / ``parse_payload`` / ...) keep operating on ``ctx``
exactly as before.

This module is kept import-light at the top level (no CommonClient / Utils) so importing it from
the world's ``__init__.py`` does not pull client-only modules in at generation time. The heavy
imports happen lazily inside the methods.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional

from .signature import BTHACK_ANCHOR_OFFSET, RDRAM_BASE, is_rdram_pointer


# --------------------------------------------------------------------------- #
# Memory adapter: presents the BTEmuLoaderClient interface backed by ctx memory
# --------------------------------------------------------------------------- #
class _CtxMemoryAdapter:
    """Adapts the EmuLoader client ``ctx`` to the ``BTEmuLoaderClient`` interface that
    ``client.state.BTHReader`` and ``client.game`` expect (read/write + pointer-chase helpers)."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx

    def read_u8(self, address: int) -> int:
        return self.ctx.read_u8(address)

    def read_u16(self, address: int) -> int:
        return self.ctx.read_u16(address)

    def read_u32(self, address: int) -> int:
        return self.ctx.read_u32(address)

    def write_u8(self, address: int, value: int) -> None:
        self.ctx.write_u8(address, value)

    def write_u16(self, address: int, value: int) -> None:
        self.ctx.write_u16(address, value)

    def write_u32(self, address: int, value: int) -> None:
        self.ctx.write_u32(address, value)

    def deref(self, address: int) -> Optional[int]:
        ptr = self.read_u32(address)
        return ptr & 0x7FFFFFFF if is_rdram_pointer(ptr) else None

    def get_anchor(self) -> Optional[int]:
        return self.deref(RDRAM_BASE + BTHACK_ANCHOR_OFFSET)

    def get_rom_version(self) -> Optional[tuple[int, int, int]]:
        anchor = self.get_anchor()
        if anchor is None:
            return None
        major = self.read_u16(RDRAM_BASE + anchor + 0x0)
        minor = self.read_u8(RDRAM_BASE + anchor + 0x2)
        patch = self.read_u8(RDRAM_BASE + anchor + 0x3)
        return (major, minor, patch)


# --------------------------------------------------------------------------- #
# ctx state setup (ports BanjoTooieContext.__init__ + the BT-specific methods)
# --------------------------------------------------------------------------- #
def _init_bt_state(ctx: Any) -> None:
    """Initialise Banjo-Tooie session state + helper methods on ``ctx``.

    Called when a transport is adopted. Resets the per-connection polling/diff state so a fresh
    emulator/bridge attachment re-syncs from scratch; safe to call again on reconnect.
    """
    from CommonClient import logger

    # Emulator-poll bookkeeping
    ctx.emu_settings_written = False
    ctx.emu_last_items_count = -1
    ctx.emu_sent_world_entrances = set()
    ctx.emu_goal_printed = False
    ctx._bt_prev_flags = None

    # Socket (EverDrive) transport bookkeeping
    if not hasattr(ctx, "n64_streams"):
        ctx.n64_streams = None
    ctx.n64_status = getattr(ctx, "n64_status", "Connection has not been initiated")
    ctx.sendSlot = False
    ctx.sync_ready = False
    ctx.awaiting_rom = False

    # Location diff tables (ED64 payload diffing)
    for name in (
        "location_table", "movelist_table", "cheatorewardslist_table", "honeybrewardslist_table",
        "treblelist_table", "stationlist_table", "jinjofamlist_table", "jinjolist_table",
        "pages_table", "honeycomb_table", "glowbo_table", "doubloon_table", "notes_table",
        "worldlist_table", "chuffy_table", "mystery_table", "roystenlist_table",
        "jiggychunks_table", "dino_kids_table", "boggy_kids_table", "alien_kids_table",
        "skivvies_table", "mr_fit_table", "bt_tickets_table", "green_relics_table", "beans_table",
        "signpost_table", "warppads_table", "silos_table", "nests_table", "jiggy_table",
    ):
        setattr(ctx, name, {})
    ctx.goggles_table = False
    ctx.roar = False
    ctx.current_map = 0

    # Links
    ctx.deathlink_enabled = getattr(ctx, "deathlink_enabled", False)
    ctx.deathlink_pending = False
    ctx.deathlink_sent_this_death = False
    ctx.deathlink_client_override = getattr(ctx, "deathlink_client_override", False)
    ctx.taglink_enabled = getattr(ctx, "taglink_enabled", False)
    ctx.pending_tag_link = False
    ctx.taglink_sent_this_tag = False
    ctx.taglink_client_override = getattr(ctx, "taglink_client_override", False)

    ctx.version_warning = getattr(ctx, "version_warning", False)
    ctx.rom_version = getattr(ctx, "rom_version", "")
    ctx.messages = getattr(ctx, "messages", {})
    ctx.startup = getattr(ctx, "startup", False)
    ctx.handled_scouts = getattr(ctx, "handled_scouts", [])
    if not hasattr(ctx, "last_death_link"):
        ctx.last_death_link = 0

    # BT-specific ctx methods the payload helpers call. Bound as closures so get_payload /
    # parse_payload keep working against `ctx` unchanged.
    def set_message(msg: Any) -> None:
        ctx.messages[len(ctx.messages) + 1] = msg

    async def update_tag_link(tag_link: bool) -> None:
        old_tags = ctx.tags.copy()
        if tag_link:
            ctx.tags.add("TagLink")
        else:
            ctx.tags -= {"TagLink"}
        if old_tags != ctx.tags and ctx.server and not ctx.server.socket.closed:
            await ctx.send_msgs([{"cmd": "ConnectUpdate", "tags": ctx.tags}])

    async def send_tag_link() -> None:
        import time
        if "TagLink" not in ctx.tags or ctx.slot is None:
            return
        if not hasattr(ctx, "instance_id"):
            ctx.instance_id = time.time()
        await ctx.send_msgs([{"cmd": "Bounce", "tags": ["TagLink"],
                              "data": {"time": time.time(), "source": ctx.instance_id, "tag": True}}])

    ctx.set_message = set_message
    ctx.update_tag_link = update_tag_link
    ctx.send_tag_link = send_tag_link
    ctx._bt_state_inited = True
    logger.debug("Banjo-Tooie handler state initialised.")


class BanjoTooieEmuHandler:
    """Duck-typed ``n64_client_handler`` for the EmuLoader client."""

    items_handling = 0b111  # full + starting inventory + own world

    # ------------------------------------------------------------------ #
    # Identification / auth
    # ------------------------------------------------------------------ #
    async def validate_rom(self, ctx: Any) -> bool:
        """Confirm the loaded ROM matches the connected world's AP version, init state."""
        from CommonClient import logger
        from ..BTClient import version

        _init_bt_state(ctx)
        loader = _CtxMemoryAdapter(ctx)
        rom_version_tuple = loader.get_rom_version()
        if rom_version_tuple is not None and rom_version_tuple[0] > 0:
            ctx.rom_version = f"{rom_version_tuple[0]}.{rom_version_tuple[1]}.{rom_version_tuple[2]}"
            if version != ctx.rom_version:
                logger.error(
                    f"ERROR: Your Patched ROM is version {ctx.rom_version}, expected {version}. "
                    "Please update to the latest version."
                )
                ctx.version_warning = True
                return False
        return True

    def wants_username_prompt(self, ctx: Any) -> bool:
        # Emulator users type their slot name; the EverDrive bridge supplies it instead.
        return ctx.transport_mode != "alternate"

    # ------------------------------------------------------------------ #
    # Server packet hooks
    # ------------------------------------------------------------------ #
    def on_package(self, ctx: Any, cmd: str, args: dict) -> None:
        from CommonClient import logger
        from ..BTClient import bt_itm_name_to_id, version
        import time

        if cmd == "Connected":
            ctx.slot_data = args.get("slot_data", {}) or {}
            slot_version = ctx.slot_data.get("custom_bt_data", {}).get("version", "N/A")
            if version != slot_version:
                logger.error(
                    "Your Banjo-Tooie AP does not match the generated world.\n"
                    f"Your version: {version} | Generated version: {slot_version}"
                )
            if ctx.rom_version and version != ctx.rom_version:
                logger.error(
                    f"ERROR: Your Patched ROM is version {ctx.rom_version}, expected {version}."
                )
            ctx.deathlink_enabled = bool(ctx.slot_data.get("options", {}).get("death_link"))
            ctx.taglink_enabled = bool(ctx.slot_data.get("options", {}).get("tag_link"))
        elif cmd == "ReceivedItems":
            if ctx.startup is False:
                for item in args["items"]:
                    player = next((n for (i, n) in ctx.player_names.items() if i == item.player), "")
                    item_name = next((n for (n, i) in bt_itm_name_to_id.items() if i == item.item), "")
                    logger.info(f"{player} sent {item_name}")
                logger.info("The above items will be sent when Banjo-Tooie is loaded.")
                ctx.startup = True

        if isinstance(args.get("data", {}), dict):
            source_name = args.get("data", {}).get("source", None)
            if not hasattr(ctx, "instance_id"):
                ctx.instance_id = time.time()
            if "TagLink" in ctx.tags and source_name != ctx.instance_id and "TagLink" in args.get("tags", []):
                ctx.pending_tag_link = getattr(ctx, "pending_tag_link", False) or hasattr(ctx, "pending_tag_link")

    def on_print_json(self, ctx: Any, args: dict) -> None:
        """Queue item-send toasts so the in-game dialog feature can surface them."""
        if args.get("type") != "ItemSend":
            return
        try:
            if not (ctx.slot_concerns_self(args["receiving"]) or ctx.slot_concerns_self(args["item"].player)):
                return
            player = ctx.player_names[int(args["data"][0]["text"])]
            to_player = player
            for idx, data in enumerate(args["data"]):
                if idx == 0:
                    continue
                if data.get("type") == "player_id":
                    to_player = ctx.player_names[int(data["text"])]
                    break
            item_name = ctx.item_names.lookup_in_slot(int(args["data"][2]["text"]))
            ctx.set_message({
                "player": player, "item": item_name,
                "item_id": int(args["data"][2]["text"]), "to_player": to_player,
            })
        except Exception:
            pass

    def on_deathlink(self, ctx: Any, data: dict) -> None:
        ctx.deathlink_pending = True

    # ------------------------------------------------------------------ #
    # Alternate transport (EverDrive 64 socket bridge on localhost:21221)
    # ------------------------------------------------------------------ #
    async def alternate_connect(self, ctx: Any) -> bool:
        from CommonClient import logger
        try:
            ctx.n64_streams = await asyncio.wait_for(
                asyncio.open_connection("localhost", 21221), timeout=10)
        except (asyncio.TimeoutError, ConnectionRefusedError, OSError, Exception):
            ctx.n64_streams = None
            return False
        _init_bt_state(ctx)
        ctx.n64_status = "Initial Connection Made"
        logger.info("Connected to the EverDrive 64 bridge. Use /n64 for status.")
        return True

    async def alternate_connected(self, ctx: Any) -> bool:
        return getattr(ctx, "n64_streams", None) is not None

    def alternate_disconnect(self, ctx: Any) -> None:
        streams = getattr(ctx, "n64_streams", None)
        if streams is not None:
            try:
                streams[1].close()
            except Exception:
                pass
        ctx.n64_streams = None

    # ------------------------------------------------------------------ #
    # Per-tick driver
    # ------------------------------------------------------------------ #
    async def game_watcher(self, ctx: Any) -> None:
        if ctx.transport_mode == "alternate":
            await self._alternate_tick(ctx)
        else:
            await self._emulator_tick(ctx)

    async def _alternate_tick(self, ctx: Any) -> None:
        """One request/response exchange with the EverDrive bridge (ports n64_sync_task body)."""
        from CommonClient import logger
        from ..BTClient import get_payload, get_slot_payload, parse_payload, script_version

        streams = getattr(ctx, "n64_streams", None)
        if streams is None:
            return
        reader, writer = streams
        msg = (get_slot_payload(ctx) if ctx.sendSlot else get_payload(ctx)).encode()
        writer.write(msg)
        writer.write(b"\n")
        try:
            await asyncio.wait_for(writer.drain(), timeout=1.5)
            data = await asyncio.wait_for(reader.readline(), timeout=10)
            data_decoded = json.loads(data.decode())
        except (asyncio.TimeoutError, ConnectionResetError, TimeoutError, OSError):
            self.alternate_disconnect(ctx)  # next tick: watcher reconnects
            ctx.n64_status = "Lost connection to N64; reconnecting..."
            return

        if data_decoded.get("getSlot", 0) is True:
            ctx.sendSlot = True
            return
        if data_decoded.get("scriptVersion", 0) < script_version:
            if not ctx.version_warning:
                logger.warning(
                    f"Your N64 bridge is older than expected (need {script_version}). "
                    "Please update; your AP connection will not be accepted."
                )
                ctx.version_warning = True
            return
        ctx.n64_status = "Connected"
        if ctx.game is not None and "jiggies" in data_decoded:
            await parse_payload(data_decoded, ctx, False)
        if not ctx.auth:
            ctx.auth = data_decoded["playerName"]
            await ctx.server_auth(False)

    async def _emulator_tick(self, ctx: Any) -> None:
        """One poll of emulator memory (ports the emu_loader_monitor_task inner body)."""
        from CommonClient import logger
        from . import state as emu_state, game as emu_game, addresses as emu_addresses
        from ..BTClient import CreateHintsParams, mumbo_tokens_loc, version

        if ctx.version_warning:
            return

        loader = _CtxMemoryAdapter(ctx)
        bth = emu_state.BTHReader(loader)

        rom_version_tuple = loader.get_rom_version()
        if rom_version_tuple is not None and rom_version_tuple[0] > 0 and ctx.version_warning is False:
            ctx.rom_version = f"{rom_version_tuple[0]}.{rom_version_tuple[1]}.{rom_version_tuple[2]}"
            if version != ctx.rom_version:
                ctx.version_warning = True
                logger.error(
                    f"ERROR: Your Patched ROM is version {ctx.rom_version}, expected {version}. "
                    "Please update to the latest version.")
                return

        # Push slot settings into RAM (the ROM refuses to boot until populated)
        expected_seed = None
        if ctx.slot_data:
            s = ctx.slot_data.get("custom_bt_data", {}).get("seed")
            if isinstance(s, int) and s:
                expected_seed = s & 0xFFFFFFFF
        settings_ptr = bth.settings_ptr()
        if expected_seed is not None and settings_ptr is not None:
            current_seed = loader.read_u32(settings_ptr + emu_game.SETTING_SEED)
            if current_seed != expected_seed:
                emu_game.write_slot_settings(loader, ctx.slot_data)
                ctx.emu_last_items_count = -1
                ctx.emu_sent_world_entrances.clear()
                ctx.emu_goal_printed = False
            ctx.emu_settings_written = True

        # Received items
        current_items_count = len(ctx.items_received) if ctx.items_received else 0
        if (ctx.emu_settings_written and current_items_count != ctx.emu_last_items_count
                and bth.items_ptr() is not None and bth.traps_ptr() is not None):
            emu_game.write_received_items(loader, ctx.items_received)
            ctx.emu_last_items_count = current_items_count

        # DeathLink
        if ctx.emu_settings_written and ctx.deathlink_enabled:
            pc_ptr = bth.pc_ptr()
            if pc_ptr is not None:
                n64_us = bth.n64_death()
                pc_us = bth.pc_death()
                if n64_us != pc_us:
                    loader.write_u8(pc_ptr + emu_state.PC_DEATH_US, n64_us & 0xFF)
                    if not ctx.deathlink_sent_this_death and ctx.server is not None:
                        ctx.deathlink_sent_this_death = True
                        await ctx.send_death()
                else:
                    ctx.deathlink_sent_this_death = False
                if ctx.deathlink_pending:
                    cur_ap = loader.read_u8(pc_ptr + emu_state.PC_DEATH_AP)
                    loader.write_u8(pc_ptr + emu_state.PC_DEATH_AP, (cur_ap + 1) & 0xFF)
                    ctx.deathlink_pending = False

        if ctx.emu_settings_written and ctx.slot_data:
            eligible = emu_game.check_world_entrances_open(loader, ctx.slot_data)
            new_entrances = [loc for loc in eligible if loc not in ctx.emu_sent_world_entrances]
            if new_entrances and ctx.server is not None:
                missing = ctx.missing_locations or set()
                to_send = [b for b in new_entrances if b in missing] if missing else list(new_entrances)
                if to_send:
                    await ctx.send_msgs([{"cmd": "LocationChecks", "locations": to_send}])
                ctx.emu_sent_world_entrances.update(new_entrances)

            emu_game.apply_hag1_open(loader, ctx.slot_data)

            if not ctx.finished_game and ctx.server is not None:
                if emu_game.check_victory(loader, ctx.slot_data):
                    await ctx.send_msgs([{"cmd": "StatusUpdate", "status": 30}])
                    ctx.finished_game = True

            # Goal-info dialog
            cur_map = bth.current_map()
            if ctx.emu_goal_printed and cur_map not in (0x158, 0x18B, 0x0):
                ctx.emu_goal_printed = False
            if not ctx.emu_goal_printed and cur_map == 0x158:
                if emu_game.read_pc_text_queue(loader) == emu_game.read_n64_text_queue(loader):
                    goal_msg = emu_game.build_goal_info_message(ctx.slot_data)
                    if goal_msg is not None:
                        text, icon = goal_msg
                        if emu_game.send_pc_dialog(loader, text, icon):
                            ctx.emu_goal_printed = True

            # Drain queued item-received toasts (one per tick)
            if ctx.messages and ctx.auth:
                options = ctx.slot_data.get("options", {}) or {}
                dialog_char = emu_game.opt(options, "dialog_character", 110)
                consumed = emu_game.drain_item_messages(loader, ctx.messages, ctx.auth, dialog_char)
                if consumed:
                    ctx.messages.pop(consumed, None)

            # Tracker map key
            if ctx.current_map != bth.current_map():
                ctx.current_map = bth.current_map()
                await ctx.send_msgs([{
                    "cmd": "Set", "key": f"Banjo_Tooie_{ctx.team}_{ctx.slot}_map",
                    "default": hex(0), "want_reply": False,
                    "operations": [{"operation": "replace", "value": hex(bth.current_map())}],
                }])

        # Location flags
        collected = emu_state.poll_all_locations(bth)
        prev = ctx._bt_prev_flags
        if prev is None:
            new_btids = [b for b, v in collected.items() if v]
        else:
            new_btids = [b for b, v in collected.items() if v and not prev.get(b, False)]

        if ctx.slot_data:
            vc = ctx.slot_data.get("options", {}).get("victory_condition")
            if vc in (1, 2, 3, 4, 6):
                new_btids = mumbo_tokens_loc(new_btids, vc)

        if new_btids and ctx.server is not None:
            missing = ctx.missing_locations or set()
            to_send = [b for b in new_btids if b in missing] if missing else list(new_btids)
            if to_send:
                await ctx.send_msgs([{"cmd": "LocationChecks", "locations": to_send}])

            signpost_btids = emu_addresses.BY_CATEGORY.get("SIGNPOSTS", {})
            if signpost_btids and ctx.slot_data:
                actual_hints = ctx.slot_data.get("custom_bt_data", {}).get("hints") or {}
                for btid in new_btids:
                    if btid not in signpost_btids:
                        continue
                    hint = actual_hints.get(str(btid))
                    if (hint is None or not hint.get("should_add_hint")
                            or hint.get("location_id") is None
                            or hint.get("location_player_id") is None):
                        continue
                    params = CreateHintsParams(hint["location_id"], hint["location_player_id"])
                    if params in ctx.handled_scouts:
                        continue
                    await ctx.send_msgs([{
                        "cmd": "CreateHints", "locations": [params.location], "player": params.player}])
                    ctx.handled_scouts.append(params)

        ctx._bt_prev_flags = collected

    # ------------------------------------------------------------------ #
    # Slash commands (exposed in the EmuLoader client via client_commands)
    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        self.client_commands = {
            "patch": self.cmd_patch,
            "autostart": self.cmd_autostart,
            "rom_path": self.cmd_rom_path,
            "patch_path": self.cmd_patch_path,
            "program_args": self.cmd_program_args,
            "n64": self.cmd_n64,
            "deathlink": self.cmd_deathlink,
            "taglink": self.cmd_taglink,
            "writesettings": self.cmd_writesettings,
        }

    def cmd_patch(self, ctx: Any, *args: str):
        """Re-run the ROM patcher."""
        from Utils import async_start
        from ..BTClient import patch_and_run
        async_start(patch_and_run(True))

    def cmd_autostart(self, ctx: Any, *args: str):
        """Configure (or disable) a program to auto-launch with the client."""
        import os
        from CommonClient import logger
        from Utils import async_start, open_filename
        from .. import BTClient as btc
        program_path = btc.bt_options.get("program_path", "")
        if program_path == "" or not os.path.isfile(program_path):
            program_path = open_filename("Select your program to automatically start", (("All Files", "*"),))
            if program_path:
                btc.bt_options.program_path = program_path
                btc.bt_options._changed = True
                logger.info(f"Autostart configured for: {program_path}")
                if not btc.program or btc.program.poll() is not None:
                    async_start(btc.patch_and_run(False))
            else:
                logger.error("No file selected...")
        else:
            btc.bt_options.program_path = ""
            btc.bt_options._changed = True
            logger.info("Autostart disabled.")

    def cmd_rom_path(self, ctx: Any, path: str = ""):
        """Set (or unset) the vanilla ROM path used for patching."""
        from CommonClient import logger
        from ..BTClient import bt_options
        bt_options.rom_path = path
        bt_options._changed = True
        logger.info("rom_path set!" if path else "rom_path unset!")

    def cmd_patch_path(self, ctx: Any, path: str = ""):
        """Set (or unset) the folder to save the patched ROM."""
        from CommonClient import logger
        from ..BTClient import bt_options
        bt_options.patch_path = path
        bt_options._changed = True
        logger.info("patch_path set!" if path else "patch_path unset!")

    def cmd_program_args(self, ctx: Any, path: str = ""):
        """Set (or unset) arguments passed to the auto-run program."""
        from CommonClient import logger
        from ..BTClient import bt_options
        bt_options.program_args = path
        bt_options._changed = True
        logger.info("program_args set!" if path else "program_args unset!")

    def cmd_n64(self, ctx: Any, *args: str):
        """Show emulator / EverDrive socket connection status."""
        from CommonClient import logger
        mode = ctx.transport_mode or "not connected"
        logger.info(f"Transport: {mode}")
        logger.info(f"Socket (EverDrive): {getattr(ctx, 'n64_status', 'n/a')}")

    def cmd_deathlink(self, ctx: Any, *args: str):
        """Toggle DeathLink from the client (overrides the slot default)."""
        from Utils import async_start
        ctx.deathlink_client_override = True
        ctx.deathlink_enabled = not ctx.deathlink_enabled
        async_start(ctx.update_death_link(ctx.deathlink_enabled), name="Update Deathlink")

    def cmd_taglink(self, ctx: Any, *args: str):
        """Toggle TagLink from the client (overrides the slot default)."""
        from Utils import async_start
        ctx.taglink_client_override = True
        ctx.taglink_enabled = not ctx.taglink_enabled
        async_start(ctx.update_tag_link(ctx.taglink_enabled), name="Update Taglink")

    def cmd_writesettings(self, ctx: Any, *args: str):
        """Push slot settings into BTHACK memory (emulator transport only)."""
        from . import game as emu_game
        if ctx.transport_mode != "emulator" or not ctx.slot_data:
            return
        if emu_game.write_slot_settings(_CtxMemoryAdapter(ctx), ctx.slot_data):
            ctx.emu_settings_written = True
