import numpy as np
from datetime import datetime
from functools import wraps
import traceback


DEBUG_ENABLED = True


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    BLUE = "\033[34m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    GRAY = "\033[90m"
    WHITE = "\033[97m"


def _to_numpy(arr):
    """Convert tensor to numpy array if needed."""
    # PyTorch tensor
    if hasattr(arr, 'detach'):
        return arr.detach().cpu().numpy()
    # TensorFlow tensor
    elif hasattr(arr, 'numpy'):
        try:
            return arr.numpy()
        except:
            return np.array(arr)
    # Already numpy or other
    return arr


def _format_array_summary(arr, max_elements=6):
    """Format numpy arrays and tensors with shape, dtype, and value preview."""
    # Get framework info
    framework = "numpy"
    device_info = ""
    
    if hasattr(arr, 'detach'):  # PyTorch
        framework = "torch"
        device_info = f", device={arr.device}"
        if arr.requires_grad:
            device_info += ", grad=True"
    elif hasattr(arr, 'numpy') and not isinstance(arr, np.ndarray):  # TensorFlow
        framework = "tensorflow"
        if hasattr(arr, 'device'):
            device_info = f", device={arr.device}"
    
    if not isinstance(arr, np.ndarray) and not hasattr(arr, 'shape'):
        return str(arr)

    # Convert to numpy for statistics
    arr_np = _to_numpy(arr)
    flat = arr_np.flatten()
    stats = []

    stats.append(f"shape={arr.shape}")
    stats.append(f"dtype={arr.dtype}{device_info}")

    if np.issubdtype(arr_np.dtype, np.number):
        if arr_np.size > 0:
            stats.append(f"range=[{arr_np.min():.4g}, {arr_np.max():.4g}]")
            stats.append(f"mean={arr_np.mean():.4g}")

    # Add value preview
    if flat.size > max_elements:
        preview = np.concatenate(
            [flat[:max_elements//2], flat[-max_elements//2:]])
        with np.printoptions(precision=4, suppress=True):
            values = f"[{flat[0]:.4g} {flat[1]:.4g} ... {flat[-2]:.4g} {flat[-1]:.4g}]"
    else:
        with np.printoptions(precision=4, suppress=True):
            values = str(flat)

    return ", ".join(stats) + f"\n      values: {values}"


def inspect(name, value, enabled=None, show_stats=True):
    """
    Inspect a variable with detailed information.

    Args:
        name: Variable name (string)
        value: Variable value (numpy array, torch tensor, tf tensor, etc.)
        enabled: Override global DEBUG_ENABLED
        show_stats: Show statistics for arrays

    Returns:
        value (unchanged, for chaining)
    """
    if enabled is None:
        enabled = DEBUG_ENABLED
    if not enabled:
        return value

    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

    # Header
    print(f"\n{C.DIM}╭─[{ts}]{C.RESET}")
    print(
        f"{C.DIM}│{C.RESET} {C.CYAN}🔍 {C.BOLD}{name}{C.RESET} "
        f"{C.DIM}({type(value).__name__}){C.RESET}"
    )
    print(f"{C.DIM}├─{C.RESET}")

    # Content
    if isinstance(value, np.ndarray) or hasattr(value, 'shape'):
        info = _format_array_summary(value)
        for line in info.split('\n'):
            print(f"{C.DIM}│{C.RESET}   {C.YELLOW}{line}{C.RESET}")

        # Show actual array with limited display
        print(f"{C.DIM}│{C.RESET}")
        arr_np = _to_numpy(value)
        with np.printoptions(precision=4, threshold=10, edgeitems=2, suppress=True):
            array_str = str(arr_np)
            for line in array_str.split('\n'):
                print(f"{C.DIM}│{C.RESET}   {C.DIM}{line}{C.RESET}")
    else:
        print(f"{C.DIM}│{C.RESET}   {C.DIM}{repr(value)}{C.RESET}")

    print(f"{C.DIM}╰─{C.RESET}")

    return value


def debug_func(enabled=None, show_time=True):
    """
    Decorator to debug function inputs and outputs.

    Args:
        enabled: Override global DEBUG_ENABLED
        show_time: Show execution time
    """
    if enabled is None:
        enabled = DEBUG_ENABLED

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)

            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            # Function entry
            print(
                f"\n{C.DIM}╔══════════════════════════════════════════════════════════{C.RESET}")
            print(
                f"{C.DIM}║{C.RESET} {C.MAGENTA}🧠 {C.BOLD}{func.__name__}(){C.RESET} {C.DIM}[{ts}]{C.RESET}")
            print(
                f"{C.DIM}╠──────────────────────────────────────────────────────────{C.RESET}")

            # Input summary
            if args or kwargs:
                print(f"{C.DIM}║{C.RESET} {C.BLUE}{C.BOLD}Inputs:{C.RESET}")
                for i, arg in enumerate(args):
                    summary = _format_array_summary(arg)
                    print(f"{C.DIM}║{C.RESET}   {C.BLUE}arg[{i}]:{C.RESET}")
                    for line in summary.split('\n'):
                        print(f"{C.DIM}║{C.RESET}      {line}")

                for k, v in kwargs.items():
                    summary = _format_array_summary(v)
                    print(f"{C.DIM}║{C.RESET}   {C.BLUE}{k}:{C.RESET}")
                    for line in summary.split('\n'):
                        print(f"{C.DIM}║{C.RESET}      {line}")
                print(
                    f"{C.DIM}╠──────────────────────────────────────────────────────────{C.RESET}")

            # Execute function
            start_time = datetime.now() if show_time else None
            try:
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                error = e
                result = None

            end_time = datetime.now() if show_time else None

            # Output summary
            if success:
                print(f"{C.DIM}║{C.RESET} {C.GREEN}{C.BOLD}Output:{C.RESET}")
                summary = _format_array_summary(result)
                for line in summary.split('\n'):
                    print(f"{C.DIM}║{C.RESET}   {line}")
            else:
                print(f"{C.DIM}║{C.RESET} {C.RED}{C.BOLD}Error: {error}{C.RESET}")
                print(f"{C.DIM}║{C.RESET} {C.RED}{traceback.format_exc()}{C.RESET}")

            # Timing info
            if show_time and start_time and end_time:
                elapsed = (end_time - start_time).total_seconds() * 1000
                print(
                    f"{C.DIM}╠──────────────────────────────────────────────────────────{C.RESET}")
                print(
                    f"{C.DIM}║{C.RESET} {C.GRAY}⏱  Execution time: {elapsed:.2f}ms{C.RESET}")

            print(
                f"{C.DIM}╚══════════════════════════════════════════════════════════{C.RESET}\n")

            if not success:
                raise error

            return result

        return wrapper
    return decorator


def assert_shape(name, arr, expected_shape, enabled=None):
    """Assert array shape matches expected, with helpful error message."""
    if enabled is None:
        enabled = DEBUG_ENABLED
    if not enabled:
        return

    if arr.shape != expected_shape:
        print(f"{C.RED}❌ Shape mismatch for {C.BOLD}{name}{C.RESET}")
        print(f"   Expected: {expected_shape}")
        print(f"   Got: {arr.shape}")
        raise ValueError(
            f"Shape mismatch for {name}: expected {expected_shape}, got {arr.shape}")


def shape_trace(*arrays, names=None, enabled=None):
    """
    Print shapes of multiple arrays/tensors in a clean table format.
    Perfect for understanding data flow through your network.

    Args:
        *arrays: Variable number of numpy arrays or tensors
        names: List of names for each array (optional)
        enabled: Override global DEBUG_ENABLED

    Example:
        shape_trace(X, W1, h, W2, y, names=['X', 'W1', 'h', 'W2', 'y'])
    """
    if enabled is None:
        enabled = DEBUG_ENABLED
    if not enabled:
        return

    if names is None:
        names = [f"arr_{i}" for i in range(len(arrays))]

    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"\n{C.CYAN}📐 Shape Trace {C.DIM}[{ts}]{C.RESET}")
    print(f"{C.DIM}{'─' * 50}{C.RESET}")

    max_name_len = max(len(name) for name in names)

    for name, arr in zip(names, arrays):
        if isinstance(arr, np.ndarray) or hasattr(arr, 'shape'):
            shape_str = str(arr.shape).ljust(20)
            dtype_str = str(arr.dtype).ljust(10)
            
            # Add device info for tensors
            device_str = ""
            if hasattr(arr, 'device'):
                device_str = f" on {arr.device}"
            
            size = np.prod(arr.shape) if hasattr(arr, 'shape') else 0
            print(f"{C.BOLD}{name.ljust(max_name_len)}{C.RESET} │ {C.YELLOW}{shape_str}{C.RESET} │ {C.GRAY}{dtype_str}{device_str}{C.RESET} │ {C.DIM}{size:,} elements{C.RESET}")
        else:
            print(
                f"{C.BOLD}{name.ljust(max_name_len)}{C.RESET} │ {C.RED}Not an array{C.RESET}")

    print(f"{C.DIM}{'─' * 50}{C.RESET}\n")


def shape_evolution(name, shapes_dict, enabled=None):
    """
    Track how a tensor's shape evolves through network layers.

    Args:
        name: Name of the tensor being tracked
        shapes_dict: OrderedDict or dict with {layer_name: shape} pairs
        enabled: Override global DEBUG_ENABLED

    Example:
        shape_evolution("activations", {
            "input": (32, 784),
            "hidden1": (32, 256),
            "hidden2": (32, 128),
            "output": (32, 10)
        })
    """
    if enabled is None:
        enabled = DEBUG_ENABLED
    if not enabled:
        return

    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(
        f"\n{C.MAGENTA}🔄 Shape Evolution: {C.BOLD}{name}{C.RESET} {C.DIM}[{ts}]{C.RESET}")
    print(f"{C.DIM}{'─' * 50}{C.RESET}")

    prev_shape = None
    for i, (layer, shape) in enumerate(shapes_dict.items()):
        arrow = "   " if i == 0 else " ↓ "

        if prev_shape and prev_shape != shape:
            # Highlight dimension changes
            changes = []
            if len(prev_shape) != len(shape):
                changes.append(f"dims: {len(prev_shape)}→{len(shape)}")
            else:
                for j, (old, new) in enumerate(zip(prev_shape, shape)):
                    if old != new:
                        changes.append(f"dim[{j}]: {old}→{new}")

            change_str = ", ".join(changes)
            print(
                f"{C.YELLOW}{arrow}{C.RESET}{C.BOLD}{layer.ljust(15)}{C.RESET} {shape} {C.DIM}({change_str}){C.RESET}")
        else:
            print(
                f"{C.GREEN}{arrow}{C.RESET}{C.BOLD}{layer.ljust(15)}{C.RESET} {shape}")

        prev_shape = shape

    print(f"{C.DIM}{'─' * 50}{C.RESET}\n")


def visualize_shape(name, arr, enabled=None):
    """
    Create ASCII art visualization of array/tensor dimensions.
    Useful for understanding tensor shapes intuitively.

    Example:
        visualize_shape("conv_output", torch.zeros((32, 64, 28, 28)))
    """
    if enabled is None:
        enabled = DEBUG_ENABLED
    if not enabled:
        return

    if not isinstance(arr, np.ndarray) and not hasattr(arr, 'shape'):
        print(f"{C.RED}Not an array or tensor{C.RESET}")
        return

    shape = arr.shape
    ndim = len(shape)
    
    # Get size info
    if isinstance(arr, np.ndarray):
        itemsize = arr.itemsize
    elif hasattr(arr, 'element_size'):  # PyTorch
        itemsize = arr.element_size()
    else:
        itemsize = 4  # default assumption

    print(f"\n{C.CYAN}📦 Shape Visualization: {C.BOLD}{name}{C.RESET}")
    print(f"{C.DIM}{'─' * 50}{C.RESET}")

    if ndim == 1:
        print(
            f"  Vector: [{C.YELLOW}{'─' * min(shape[0], 40)}{C.RESET}] ({shape[0]} elements)")

    elif ndim == 2:
        rows, cols = shape
        print(f"  Matrix ({rows}×{cols}):")
        print(f"    {C.YELLOW}┌{' ' * min(cols, 38)}┐{C.RESET}")
        for _ in range(min(rows, 5)):
            print(f"    {C.YELLOW}│{' ' * min(cols, 38)}│{C.RESET}")
        if rows > 5:
            print(f"    {C.DIM}  ... {rows - 5} more rows ...{C.RESET}")
        print(f"    {C.YELLOW}└{' ' * min(cols, 38)}┘{C.RESET}")

    elif ndim == 3:
        d0, d1, d2 = shape
        print(f"  3D Tensor ({d0}×{d1}×{d2}):")
        print(f"    {C.MAGENTA}Batch/Depth: {d0}{C.RESET}")
        print(f"    {C.YELLOW}Height: {d1}{C.RESET}")
        print(f"    {C.CYAN}Width: {d2}{C.RESET}")
        print(f"    Visualized as {d0} matrices of {d1}×{d2}")

    elif ndim == 4:
        d0, d1, d2, d3 = shape
        print(f"  4D Tensor ({d0}×{d1}×{d2}×{d3}):")
        print(f"    {C.MAGENTA}Batch: {d0}{C.RESET}")
        print(f"    {C.YELLOW}Channels: {d1}{C.RESET}")
        print(f"    {C.CYAN}Height: {d2}{C.RESET}")
        print(f"    {C.GREEN}Width: {d3}{C.RESET}")
        print(f"    Common in CNNs: (batch, channels, height, width)")

    else:
        print(f"  {ndim}D Tensor: {shape}")
        total_elements = np.prod(shape)
        print(f"  Total elements: {total_elements:,}")
        print(
            f"  Memory: ~{total_elements * itemsize / 1024 / 1024:.2f} MB")

    print(f"{C.DIM}{'─' * 50}{C.RESET}\n")