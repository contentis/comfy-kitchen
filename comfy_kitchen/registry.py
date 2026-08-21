import threading
from collections.abc import Callable, Mapping
from contextlib import contextmanager

from .exceptions import BackendNotFoundError


class UnsupportedCall(Exception):  # noqa: N818
    """A backend lowering cannot execute this call exactly."""


class BackendRegistry:
    def __init__(self):
        self._backends = {}  # {name: backend_module}
        self._capabilities = {}  # {name: set of function names}
        self._lowerings = {}  # {(backend_name, op_name): lowering}
        self._priority = ["cuda", "triton", "eager"]
        self._disabled = set()  # Manually disabled backends
        self._unavailable = {}  # {name: error_message} for failed backends
        self._lock = threading.Lock()
        self._generation = 0
        self._thread_local = threading.local()

    def register(
        self,
        name: str,
        module,
        lowerings: Mapping[str, Callable] | None = None,
    ):
        """Register a backend with its capabilities and constraints.

        Args:
            name: Backend name (e.g., "cuda", "eager", "triton")
            module: The backend module containing implementations
            lowerings: Implementation-driven lowerings for migrated operations
        """
        with self._lock:
            self._backends[name] = module
            if lowerings is None:
                lowerings = {
                    op: (lambda *_args, _runner=getattr(module, op), **_kwargs: _runner)
                    for op in module.__all__
                }
            self._capabilities[name] = set(lowerings)
            installed = dict(lowerings)
            for key in tuple(self._lowerings):
                if key[0] == name:
                    del self._lowerings[key]
            for op_name, lowering in installed.items():
                self._lowerings[(name, op_name)] = lowering
            self._unavailable.pop(name, None)
            self._generation += 1

    def mark_unavailable(self, name: str, reason: str):
        """Mark a backend as unavailable with a reason.

        Args:
            name: Backend name
            reason: Why the backend is unavailable (e.g., "ImportError: ...")
        """
        with self._lock:
            self._unavailable[name] = reason
            self._generation += 1

    def disable(self, backend_name: str):
        """Manually disable a backend.

        Args:
            backend_name: Name of backend to disable
        """
        with self._lock:
            self._disabled.add(backend_name)
            self._generation += 1

    def enable(self, backend_name: str):
        """Re-enable a previously disabled backend.

        Args:
            backend_name: Name of backend to enable
        """
        with self._lock:
            self._disabled.discard(backend_name)
            self._generation += 1

    def set_priority(self, priority_list: list[str]):
        """Set backend selection priority order.

        Args:
            priority_list: List of backend names in priority order
        """
        with self._lock:
            self._priority = list(priority_list)
            self._generation += 1

    def dispatch_token(self) -> tuple[int, str | None]:
        """Cheap cache token for implementation-driven dispatch."""
        return self._generation, getattr(self._thread_local, "backend_override", None)

    def iter_lowerings(self, op_name: str):
        """Yield explicitly registered lowerings in effective backend order."""
        override = getattr(self._thread_local, "backend_override", None)
        names = ([override] if override else []) + self._priority
        seen = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            lowering = self._lowerings.get((name, op_name))
            if self.is_available(name) and lowering is not None:
                yield name, lowering

    def is_available(self, backend_name: str) -> bool:
        """Check if backend is available and not disabled.

        Args:
            backend_name: Name of backend to check

        Returns:
            True if backend is registered and not disabled
        """
        return backend_name in self._backends and backend_name not in self._disabled

    def list_backends(self) -> dict:
        """Return dict of all backends with their status.

        Returns:
            Dict mapping backend names to status info:
            {
                "backend_name": {
                    "available": bool,
                    "disabled": bool,
                    "unavailable_reason": str or None,
                    "capabilities": list[str]
                }
            }
        """
        result = {}
        all_backend_names = (
            set(self._priority) | set(self._backends.keys()) | set(self._unavailable.keys())
        )

        for name in all_backend_names:
            capabilities = self._capabilities.get(name, set()) | {
                op_name for backend_name, op_name in self._lowerings if backend_name == name
            }
            result[name] = {
                "available": name in self._backends,
                "disabled": name in self._disabled,
                "unavailable_reason": self._unavailable.get(name),
                "capabilities": sorted(capabilities),
            }

        return result

    @contextmanager
    def use_backend(self, backend_name: str):
        """Context manager to temporarily use a specific backend.

        Args:
            backend_name: Name of backend to use in this context

        Example:
            with registry.use_backend("eager"):
                result = some_function()
        """
        # Validate backend exists and is available
        if not self.is_available(backend_name):
            if backend_name in self._unavailable:
                reason = self._unavailable[backend_name]
                raise BackendNotFoundError(backend_name, reason)
            elif backend_name in self._disabled:
                raise BackendNotFoundError(backend_name, "disabled")
            else:
                raise BackendNotFoundError(backend_name, "not registered")

        # Store previous override (in case of nested contexts)
        previous = getattr(self._thread_local, "backend_override", None)
        self._thread_local.backend_override = backend_name

        try:
            yield
        finally:
            # Restore previous state
            if previous is None:
                delattr(self._thread_local, "backend_override")
            else:
                self._thread_local.backend_override = previous


registry = BackendRegistry()
