"""Standalone NeoPixel LED test for the LiteWing flight stabilizer shield.

Provides a small GUI to connect to the drone, set colours, clear LEDs, and toggle
blinking. All commands are echoed to both the console and the GUI log window.
"""

import threading
import time
import tkinter as tk
from tkinter import ttk

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# UDP URI for the drone. Change this if your drone's IP is different.
DRONE_URI = "udp://192.168.43.42"

# NeoPixel CRTP port & channels
# CRTP port 0x09 (9) is used for NeoPixel commands on LiteWing
# Channels defined here match the firmware's `neopixel_crtp.c` implementation:
#  - 0: SET_PIXEL (payload: idx,r,g,b) - index 0..N-1 for a single pixel. 0xFF = broadcast / "set all".
#  - 1: SHOW (payload: none) - commit the current buffer to the LEDs (build RMT items and send).
#  - 2: CLEAR (payload: none) - zero-out the buffer and clear LEDs.
#  - 3: BLINK (payload: enable, on_ms_hi, on_ms_lo, off_ms_hi, off_ms_lo) - start/stop blink.
CRTP_PORT_NEOPIXEL = 0x09
NEOPIXEL_CHANNEL_SET_PIXEL = 0x00
NEOPIXEL_CHANNEL_SHOW = 0x01
NEOPIXEL_CHANNEL_CLEAR = 0x02
NEOPIXEL_CHANNEL_BLINK = 0x03
NP_SEND_RETRIES = 3
NP_PACKET_DELAY = 0.02
NP_LINK_SETUP_DELAY = 0.1


def _send_crtp_with_fallback(cf: Crazyflie, port: int, channel: int, payload: bytes) -> None:
    header = ((port & 0x0F) << 4) | (channel & 0x0F)

    class _Packet:
        def __init__(self, header_value: int, data: bytes):
            self.header = header_value
            self.data = data
            try:
                self.datat = tuple(data)
            except Exception:
                self.datat = tuple()

        def is_data_size_valid(self) -> bool:
            return len(self.data) <= 30

        @property
        def size(self) -> int:
            return len(self.data)

        def raw(self) -> bytes:
            return bytes([self.header]) + self.data

    # Build a packet-like object that adapts to different cflib/crazyflie API versions.
    # Some cflib versions expect a packet object with fields `header` and `data` and
    # a `datat` tuple; others expect raw bytes. This wrapper tries the different
    # send methods in turn to maximize compatibility across platforms and cflib
    # versions used by the LiteWing project.
    packet = _Packet(header, payload)

    # 1) Try Crazyflie.send_packet (some cflib versions provide this on Crazyflie instance)
    try:
        send_packet = getattr(cf, "send_packet", None)
        if callable(send_packet):
            send_packet(packet)
            return
    except Exception:  # noqa: BLE001
        pass

    # 2) Try the low-level link object (some cflib versions put send_packet on cf._link/link)
    try:
        link = getattr(cf, "_link", None) or getattr(cf, "link", None)
        if link is not None and callable(getattr(link, "send_packet", None)):
            link.send_packet(packet)
            return
    except Exception:  # noqa: BLE001
        pass

    # 3) Fallback: cflib.crtp.send_packet (try object first, then raw bytes)
    try:
        from cflib import crtp as _crtp  # Local import to avoid global dependency

        sendp = getattr(_crtp, "send_packet", None)
        if callable(sendp):
            # cflib expects either packets with .raw() or raw bytes
            try:
                sendp(packet)
                return
            except Exception:
                try:
                    sendp(packet.raw())
                    return
                except Exception:
                    pass
    except Exception:  # noqa: BLE001
        pass

    raise RuntimeError("Unable to send CRTP NeoPixel packet")


def np_set_pixel(cf: Crazyflie, index: int, r: int, g: int, b: int) -> None:
    # Build SET_PIXEL payload (index, R, G, B)
    # The index is a single byte, 0..N-1 for addressable pixels; 0xFF is a broadcast
    # value used by `set_all` to set a single color across all pixels.
    payload = bytes([index & 0xFF, r & 0xFF, g & 0xFF, b & 0xFF])
    print(f"[NeoPixel] Sending SET_PIXEL payload: {list(payload)}")
    _send_crtp_with_fallback(cf, CRTP_PORT_NEOPIXEL, NEOPIXEL_CHANNEL_SET_PIXEL, payload)


def np_set_all(cf: Crazyflie, r: int, g: int, b: int) -> None:
    # Build SET_ALL payload: index=0xFF signals the firmware to set all pixels.
    # This is how the firmware distinguishes between setting a specific pixel
    # and a broadcast 'set all' operation using the same channel value.
    payload = bytes([0xFF, r & 0xFF, g & 0xFF, b & 0xFF])
    print(f"[NeoPixel] Sending SET_ALL payload: {list(payload)}")
    _send_crtp_with_fallback(cf, CRTP_PORT_NEOPIXEL, NEOPIXEL_CHANNEL_SET_PIXEL, payload)




def np_clear(cf: Crazyflie) -> None:
    # Send the CLEAR command which zeros the buffer and commits (calls neopixelShow())
    _send_crtp_with_fallback(cf, CRTP_PORT_NEOPIXEL, NEOPIXEL_CHANNEL_CLEAR, b"")


def np_show(cf: Crazyflie) -> None:
    """Tell the NeoPixel controller to commit previously set/pending colours.

    Some implementations require SET_PIXEL to be followed by SHOW to update the LEDs
    (this is how `neopixel_control.py` and other scripts use it). Blink commands
    generally act as an effect and may update without SHOW, but SET/SET_ALL
    requires SHOW to take effect.
    """
    print("[NeoPixel] Sending SHOW command")
    # SHOW tells the firmware to build the low-level RMT items from the current
    # pixel buffer and send them over the RMT (timed output) driver onto the
    # GPIO pin. Without SHOW the pixel buffer is just updated in RAM and not
    # reflected on the LEDs until a SHOW occurs.
    _send_crtp_with_fallback(cf, CRTP_PORT_NEOPIXEL, NEOPIXEL_CHANNEL_SHOW, b"")


def np_start_blink(cf: Crazyflie, on_ms: int = 500, off_ms: int = 500) -> None:
    payload = bytes([
        1,
        (on_ms >> 8) & 0xFF,
        on_ms & 0xFF,
        (off_ms >> 8) & 0xFF,
        off_ms & 0xFF,
    ])
    print(f"[NeoPixel] Sending BLINK payload: {list(payload)}")
    # BLINK uses a 5-bytes payload: enable (1/0), on_ms (2-bytes big-endian), off_ms (2-bytes big-endian)
    # The firmware will start a FreeRTOS timer to toggle output on/off as appropriate.
    _send_crtp_with_fallback(cf, CRTP_PORT_NEOPIXEL, NEOPIXEL_CHANNEL_BLINK, payload)


def np_stop_blink(cf: Crazyflie) -> None:
    print("[NeoPixel] Sending STOP BLINK command")
    _send_crtp_with_fallback(cf, CRTP_PORT_NEOPIXEL, NEOPIXEL_CHANNEL_BLINK, bytes([0, 0, 0, 0, 0]))


def _try_send_with_retries(cf: Crazyflie, func, *args, retries: int = NP_SEND_RETRIES, logger=None) -> bool:
    last_exc: Exception | None = None
    # Reliability: try sending packets multiple times to handle transient link issues.
    for attempt in range(1, retries + 1):
        try:
            func(cf, *args)
            time.sleep(NP_PACKET_DELAY)
            return True
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if logger:
                logger(f"Attempt {attempt} failed: {exc}")
            time.sleep(NP_PACKET_DELAY)
    if logger:
        logger(f"Command failed after {retries} retries: {last_exc}")
    return False


class NeoPixelApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LiteWing NeoPixel Test")
        self.root.geometry("760x520")

        self.status_var = tk.StringVar(value="Status: Disconnected")
        self.blinking = False
        self.scf: SyncCrazyflie | None = None
        self.cf: Crazyflie | None = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=6)

        ttk.Button(control_frame, text="Connect", command=self.connect, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Disconnect", command=self.disconnect, width=12).pack(side=tk.LEFT, padx=5)
        ttk.Label(control_frame, textvariable=self.status_var, font=("Arial", 11, "bold"), foreground="blue").pack(side=tk.LEFT, padx=20)

        # Colour controls: three spinboxes for RGB values, plus a spinbox for
        # the pixel index (use -1 for broadcast to all pixels). An Auto SHOW
        # checkbox and a manual Show button let you control when the buffer
        # gets committed to the LEDs.
        spin_frame = ttk.LabelFrame(self.root, text="Colour Controls")
        spin_frame.pack(fill=tk.X, padx=10, pady=6)

        self.r_var = tk.IntVar(value=255)
        self.g_var = tk.IntVar(value=255)
        self.b_var = tk.IntVar(value=255)
        self.pixel_index_var = tk.IntVar(value=-1)
        self.auto_show_var = tk.BooleanVar(value=True)

        for label_text, var in (("R", self.r_var), ("G", self.g_var), ("B", self.b_var)):
            frame = tk.Frame(spin_frame)
            frame.pack(side=tk.LEFT, padx=6)
            ttk.Label(frame, text=f"{label_text}:").pack(side=tk.LEFT)
            ttk.Spinbox(frame, from_=0, to=255, textvariable=var, width=5).pack(side=tk.LEFT)

        index_frame = tk.Frame(spin_frame)
        index_frame.pack(side=tk.LEFT, padx=6)
        ttk.Label(index_frame, text="Pixel index (-1=all):").pack(side=tk.LEFT)
        ttk.Spinbox(index_frame, from_=-1, to=255, textvariable=self.pixel_index_var, width=6).pack(side=tk.LEFT)
        ttk.Checkbutton(index_frame, text="Auto SHOW", variable=self.auto_show_var).pack(side=tk.LEFT, padx=6)

        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=6)

        # Buttons: Set Colour sets the selected pixel or all pixels; Clear clears
        # the buffer; Blink starts a blink effect; Show will commit pending changes.
        ttk.Button(button_frame, text="Set Colour", command=self.set_colour, width=18).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_leds, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Blink", command=self.start_blink, width=14).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Show", command=self._manual_show, width=10).pack(side=tk.LEFT, padx=5)

        log_frame = ttk.LabelFrame(self.root, text="Command Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        self.log_list = tk.Listbox(log_frame, font=("Consolas", 10))
        self.log_list.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def connect(self) -> None:
        if self.cf is not None:
            self._log("Already connected")
            return

        def worker() -> None:
            self._set_status("Status: Connecting...")
            try:
                cflib.crtp.init_drivers(enable_debug_driver=False)
                # Use SyncCrazyflie to open and manage the Crazyflie link in a
                # worker thread — this avoids blocking the GUI main loop.
                scf = SyncCrazyflie(DRONE_URI, cf=Crazyflie(rw_cache="./cache"))
                scf.open_link()
                time.sleep(NP_LINK_SETUP_DELAY)
                self.scf = scf
                self.cf = scf.cf
                self._set_status("Status: Connected")
                self._log("Connected to drone")
            except Exception as exc:  # noqa: BLE001
                self._set_status("Status: Error - see console")
                self._log(f"Connection failed: {exc}")
                print(f"[NeoPixel] Connection failed: {exc}")
                self.scf = None
                self.cf = None

        threading.Thread(target=worker, daemon=True).start()

    def disconnect(self) -> None:
        if self.scf is None:
            self._log("Not connected")
            return

        def worker() -> None:
            try:
                if self.cf is not None and self.blinking:
                    _try_send_with_retries(self.cf, np_stop_blink, logger=self._log)
                if self.scf is not None:
                    self.scf.close_link()
            except Exception as exc:  # noqa: BLE001
                self._log(f"Disconnect error: {exc}")
            finally:
                self.scf = None
                self.cf = None
                self.blinking = False
                self._set_status("Status: Disconnected")
                self._log("Disconnected")

        threading.Thread(target=worker, daemon=True).start()

    def set_colour(self) -> None:
        cf = self.cf
        if cf is None:
            self._log("Set colour requested without connection")
            return
        # Get RGB values clamped to 0..255 and pack them into a bytes payload
        # via the helper functions defined earlier.
        r, g, b = self._clamp_rgb()
        pixel_index = self.pixel_index_var.get()
        self._log(f"Pixel index var raw value: {pixel_index} (type: {type(pixel_index)})")
        # If index < 0 use SET_ALL, otherwise use SET_PIXEL.
        if pixel_index < 0:
            ok = _try_send_with_retries(cf, np_set_all, r, g, b, logger=self._log)
            command = "Set all"
            if ok:
                # Commit the pixel updates (SHOW) if Auto SHOW is enabled.
                if self.auto_show_var.get():
                    _try_send_with_retries(cf, np_show, logger=self._log)
        else:
            ok = _try_send_with_retries(cf, np_set_pixel, pixel_index, r, g, b, logger=self._log)
            if ok:
                # Commit the pixel updates (SHOW) if Auto SHOW is enabled.
                if self.auto_show_var.get():
                    _try_send_with_retries(cf, np_show, logger=self._log)
            command = f"Set pixel {pixel_index}"
        if ok:
            self._log(f"{command} to RGB ({r}, {g}, {b})")



    def clear_leds(self) -> None:
        cf = self.cf
        if cf is None:
            self._log("Clear requested without connection")
            return
        if _try_send_with_retries(cf, np_clear, logger=self._log):
            self._log("Cleared LEDs")
        if self.blinking:
            if _try_send_with_retries(cf, np_stop_blink, logger=self._log):
                self._log("Stopped blinking")
            self.blinking = False

    def _manual_show(self) -> None:
        cf = self.cf
        if cf is None:
            self._log("Show requested without connection")
            return
        if _try_send_with_retries(cf, np_show, logger=self._log):
            self._log("Show (commit) sent")

    def start_blink(self) -> None:
        cf = self.cf
        if cf is None:
            self._log("Blink requested without connection")
            return
        r, g, b = self._clamp_rgb()
        pixel_index = self.pixel_index_var.get()
        if pixel_index < 0:
            if _try_send_with_retries(cf, np_set_all, r, g, b, logger=self._log):
                # Commit the pixel updates before starting the blink effect
                _try_send_with_retries(cf, np_show, logger=self._log)
                if _try_send_with_retries(cf, np_start_blink, logger=self._log):
                    self._log(f"Started blinking all with RGB ({r}, {g}, {b})")
                    self.blinking = True
        else:
            if _try_send_with_retries(cf, np_set_pixel, pixel_index, r, g, b, logger=self._log):
                # Commit the pixel updates before starting the blink effect
                _try_send_with_retries(cf, np_show, logger=self._log)
                if _try_send_with_retries(cf, np_start_blink, logger=self._log):
                    self._log(f"Started blinking pixel {pixel_index} with RGB ({r}, {g}, {b})")
                    self.blinking = True

    def _clamp_rgb(self) -> tuple[int, int, int]:
        r = max(0, min(255, self.r_var.get()))
        g = max(0, min(255, self.g_var.get()))
        b = max(0, min(255, self.b_var.get()))
        self.r_var.set(r)
        self.g_var.set(g)
        self.b_var.set(b)
        return r, g, b

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        print(f"[NeoPixel] {message}")
        self.log_list.insert(tk.END, entry)
        self.log_list.yview_moveto(1.0)

    def _on_close(self) -> None:
        self.disconnect()
        self.root.after(300, self.root.destroy)


def main() -> None:
    root = tk.Tk()
    app = NeoPixelApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
