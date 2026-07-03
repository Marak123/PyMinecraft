"""Frame-coherent input snapshot.

The window layer feeds GLFW callbacks into this object; gameplay code only
ever reads from it.  ``begin_frame`` must be called once per frame *before*
polling events so edge-triggered state (pressed this frame) works.
"""

from __future__ import annotations


class InputState:
    def __init__(self) -> None:
        self._down: set[int] = set()
        self._pressed: set[int] = set()
        self._buttons_down: set[int] = set()
        self._buttons_pressed: set[int] = set()
        self.mouse_dx: float = 0.0
        self.mouse_dy: float = 0.0
        self.scroll_dy: float = 0.0
        # First mouse delta after (re)capturing the cursor is garbage on
        # some platforms — the window layer arms this flag to swallow it.
        self._skip_next_mouse_delta: bool = False
        self._last_cursor: tuple[float, float] | None = None
        self.cursor_pos: tuple[float, float] = (0.0, 0.0)  # window coords (UI)

    # -- frame lifecycle ---------------------------------------------------
    def begin_frame(self) -> None:
        self._pressed.clear()
        self._buttons_pressed.clear()
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        self.scroll_dy = 0.0

    # -- callbacks (window layer) -------------------------------------------
    def on_key(self, key: int, pressed: bool) -> None:
        if pressed:
            if key not in self._down:
                self._pressed.add(key)
            self._down.add(key)
        else:
            self._down.discard(key)

    def on_button(self, button: int, pressed: bool) -> None:
        if pressed:
            if button not in self._buttons_down:
                self._buttons_pressed.add(button)
            self._buttons_down.add(button)
        else:
            self._buttons_down.discard(button)

    def on_cursor(self, x: float, y: float) -> None:
        self.cursor_pos = (x, y)
        if self._last_cursor is None or self._skip_next_mouse_delta:
            self._skip_next_mouse_delta = False
            self._last_cursor = (x, y)
            return
        lx, ly = self._last_cursor
        self.mouse_dx += x - lx
        self.mouse_dy += y - ly
        self._last_cursor = (x, y)

    def on_scroll(self, dy: float) -> None:
        self.scroll_dy += dy

    def reset_mouse_tracking(self) -> None:
        """Call after capturing/releasing the cursor to avoid a view jump."""
        self._skip_next_mouse_delta = True
        self._last_cursor = None

    def release_all(self) -> None:
        """Drop held state (e.g. when the window loses focus)."""
        self._down.clear()
        self._buttons_down.clear()

    # -- queries (gameplay layer) -------------------------------------------
    def is_down(self, key: int) -> bool:
        return key in self._down

    def was_pressed(self, key: int) -> bool:
        return key in self._pressed

    def is_button_down(self, button: int) -> bool:
        return button in self._buttons_down

    def was_button_pressed(self, button: int) -> bool:
        return button in self._buttons_pressed
