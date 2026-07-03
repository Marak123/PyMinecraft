"""GLFW window wrapper: context creation, event pump, cursor capture.

This is the only module in the engine that talks to GLFW directly.  It owns
the OpenGL context and forwards raw input events into an
:class:`~engine.input.InputState`.
"""

from __future__ import annotations

import glfw
import moderngl

from engine.core.log import get_logger
from engine.input import InputState

_log = get_logger("window")


class WindowError(RuntimeError):
    pass


class Window:
    def __init__(
        self,
        width: int,
        height: int,
        title: str,
        *,
        vsync: bool = True,
        fullscreen: bool = False,
    ) -> None:
        if not glfw.init():
            raise WindowError("GLFW initialisation failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.SAMPLES, 0)

        monitor = glfw.get_primary_monitor() if fullscreen else None
        if monitor is not None:
            mode = glfw.get_video_mode(monitor)
            width, height = mode.size.width, mode.size.height

        self.handle = glfw.create_window(width, height, title, monitor, None)
        if not self.handle:
            glfw.terminate()
            raise WindowError("Window creation failed (OpenGL 3.3 required)")

        glfw.make_context_current(self.handle)
        glfw.swap_interval(1 if vsync else 0)

        self.ctx: moderngl.Context = moderngl.create_context()
        self.input = InputState()
        self.focused = True
        self._cursor_captured = False
        self._vsync = vsync

        glfw.set_key_callback(self.handle, self._on_key)
        glfw.set_mouse_button_callback(self.handle, self._on_button)
        glfw.set_cursor_pos_callback(self.handle, self._on_cursor)
        glfw.set_scroll_callback(self.handle, self._on_scroll)
        glfw.set_window_focus_callback(self.handle, self._on_focus)

        if glfw.raw_mouse_motion_supported():
            glfw.set_input_mode(self.handle, glfw.RAW_MOUSE_MOTION, glfw.TRUE)

        _log.info(
            "OpenGL %s | %s", self.ctx.info.get("GL_VERSION"), self.ctx.info.get("GL_RENDERER")
        )

    # -- GLFW callbacks ------------------------------------------------------
    def _on_key(self, _win, key: int, _scan: int, action: int, _mods: int) -> None:
        if action == glfw.PRESS:
            self.input.on_key(key, True)
        elif action == glfw.RELEASE:
            self.input.on_key(key, False)

    def _on_button(self, _win, button: int, action: int, _mods: int) -> None:
        if action == glfw.PRESS:
            self.input.on_button(button, True)
        elif action == glfw.RELEASE:
            self.input.on_button(button, False)

    def _on_cursor(self, _win, x: float, y: float) -> None:
        self.input.on_cursor(x, y)

    def _on_scroll(self, _win, _dx: float, dy: float) -> None:
        self.input.on_scroll(dy)

    def _on_focus(self, _win, focused: int) -> None:
        self.focused = bool(focused)
        if not self.focused:
            self.input.release_all()

    # -- public API ------------------------------------------------------------
    @property
    def framebuffer_size(self) -> tuple[int, int]:
        return glfw.get_framebuffer_size(self.handle)

    @property
    def should_close(self) -> bool:
        return bool(glfw.window_should_close(self.handle))

    def request_close(self) -> None:
        glfw.set_window_should_close(self.handle, True)

    def set_fullscreen(self, fullscreen: bool) -> None:
        """Runtime fullscreen toggle; windowed mode restores 1280x720 centred."""
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        if fullscreen:
            glfw.set_window_monitor(
                self.handle, monitor, 0, 0, mode.size.width, mode.size.height,
                mode.refresh_rate,
            )
        else:
            w, h = 1280, 720
            x = (mode.size.width - w) // 2
            y = (mode.size.height - h) // 2
            glfw.set_window_monitor(self.handle, None, x, y, w, h, 0)
        # set_window_monitor resets the swap interval on some drivers.
        glfw.swap_interval(1 if self._vsync else 0)

    def set_vsync(self, vsync: bool) -> None:
        self._vsync = vsync
        glfw.swap_interval(1 if vsync else 0)

    def capture_cursor(self, captured: bool) -> None:
        if captured == self._cursor_captured:
            return
        mode = glfw.CURSOR_DISABLED if captured else glfw.CURSOR_NORMAL
        glfw.set_input_mode(self.handle, glfw.CURSOR, mode)
        self.input.reset_mouse_tracking()
        self._cursor_captured = captured

    @property
    def cursor_captured(self) -> bool:
        return self._cursor_captured

    def poll(self) -> None:
        self.input.begin_frame()
        glfw.poll_events()

    def swap(self) -> None:
        glfw.swap_buffers(self.handle)

    def close(self) -> None:
        glfw.terminate()
