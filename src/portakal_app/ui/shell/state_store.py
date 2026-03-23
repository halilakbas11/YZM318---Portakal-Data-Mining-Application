from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from portakal_app.models import AppState


class AppStateStore(QObject):
    stateChanged = Signal(object)

    def __init__(self, initial_state: AppState | None = None) -> None:
        super().__init__()
        self._state = initial_state or AppState()

    @property
    def state(self) -> AppState:
        return self._state

    def update(self, **changes: object) -> AppState:
        self._state = self._state.with_updates(**changes)
        self.stateChanged.emit(self._state)
        return self._state
