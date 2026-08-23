import inspect
import threading
from collections.abc import Iterable
from contextlib import contextmanager

from .exceptions import BackendNotFoundError


def _parameter_contract(function):
    return tuple(
        (name, parameter.kind, parameter.default)
        for name, parameter in inspect.signature(function).parameters.items()
    )


def _prepare_implementations(name, module, declared, eager):
    installed = set(declared)
    if eager is not None:
        installed &= set(eager.__all__)
    implementations = {operation: getattr(module, operation) for operation in installed}
    if eager is not None:
        for operation, implementation in implementations.items():
            if _parameter_contract(implementation) != _parameter_contract(getattr(eager, operation)):
                raise TypeError(f"{name}.{operation} does not match the eager signature")
    return installed, implementations


class UnsupportedCall(Exception):  # noqa: N818
    """Reject a call before the implementation performs any observable work."""


class BackendRegistry:
    def __init__(self):
        self._backends = {}  # {name: backend_module}
        self._capabilities = {}  # {name: set of function names}
        self._declared_operations = {}  # {name: requested function names}
        self._implementations = {}  # {(backend_name, op_name): function}
        self._device_types = {}  # {backend_name: supported tensor device types}
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
        operations: Iterable[str] | None = None,
        device_types: Iterable[str] = (),
    ):
        """Register backend implementations with their supported device types.

        Eager defines the shared operation surface and function signatures. Its
        registration reconciles backends that were imported before it.
        """
        declared = set(module.__all__ if operations is None else operations)
        with self._lock:
            eager = module if name == "eager" else self._backends.get("eager")
            installed, implementations = _prepare_implementations(
                name, module, declared, None if name == "eager" else eager
            )
            updates = {
                name: (module, declared, installed, implementations, frozenset(device_types))
            }

            if name == "eager":
                for backend_name, backend_module in self._backends.items():
                    if backend_name == "eager":
                        continue
                    backend_declared = self._declared_operations[backend_name]
                    backend_installed, backend_implementations = _prepare_implementations(
                        backend_name, backend_module, backend_declared, module
                    )
                    updates[backend_name] = (
                        backend_module,
                        backend_declared,
                        backend_installed,
                        backend_implementations,
                        self._device_types[backend_name],
                    )

            backends = dict(self._backends)
            declared_operations = dict(self._declared_operations)
            capabilities = dict(self._capabilities)
            registered = dict(self._implementations)
            registered_device_types = dict(self._device_types)
            unavailable = dict(self._unavailable)
            for backend_name, update in updates.items():
                backend_module, backend_declared, backend_installed, backend_impls, devices = update
                backends[backend_name] = backend_module
                declared_operations[backend_name] = backend_declared
                capabilities[backend_name] = backend_installed
                registered_device_types[backend_name] = devices
                registered = {key: value for key, value in registered.items() if key[0] != backend_name}
                registered.update(
                    {(backend_name, operation): implementation for operation, implementation in backend_impls.items()}
                )
                unavailable.pop(backend_name, None)

            self._backends = backends
            self._declared_operations = declared_operations
            self._capabilities = capabilities
            self._implementations = registered
            self._device_types = registered_device_types
            self._unavailable = unavailable
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

    def iter_implementations(self, op_name: str):
        """Yield implementations in effective backend order."""
        override = getattr(self._thread_local, "backend_override", None)
        names = ([override] if override else []) + self._priority
        seen = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            implementation = self._implementations.get((name, op_name))
            if self.is_available(name) and implementation is not None:
                yield name, implementation, self._device_types[name]

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
            capabilities = self._capabilities.get(name, set())
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
