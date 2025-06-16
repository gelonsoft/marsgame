import platform
import time

class ElapsedCpuTimer:
    """
    Timer class for measuring CPU and wall-clock time.
    """
    
    def __init__(self):
        self._os_win = platform.system() == "Windows"
        self._old_time = self._get_time()
        self._max_time = 0
        self._n_iters = 0
    
    def reset(self) -> None:
        """Reset the timer"""
        self._old_time = self._get_time()
        self._n_iters = 0
    
    def elapsed(self) -> int:
        """Elapsed time in nanoseconds"""
        return self._get_time() - self._old_time
    
    def elapsed_nanos(self) -> int:
        """Elapsed time in nanoseconds"""
        return self.elapsed()
    
    def elapsed_millis(self) -> int:
        """Elapsed time in milliseconds"""
        return self.elapsed() // 1_000_000
    
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds"""
        return self.elapsed_millis() / 1000.0
    
    def elapsed_minutes(self) -> float:
        """Elapsed time in minutes"""
        return self.elapsed_seconds() / 60.0
    
    def elapsed_hours(self) -> float:
        """Elapsed time in hours"""
        return self.elapsed_minutes() / 60.0
    
    def set_max_time_millis(self, time: int) -> None:
        """Set maximum time in milliseconds"""
        self._max_time = time * 1_000_000
    
    def remaining_time_millis(self) -> int:
        """Remaining time in milliseconds"""
        diff = self._max_time - self.elapsed()
        return diff // 1_000_000
    
    def exceeded_max_time(self) -> bool:
        """Check if maximum time has been exceeded"""
        return self.elapsed() > self._max_time
    
    def enough_budget_iteration(self, break_ms: int = 0) -> bool:
        """
        Check if there's enough budget for another iteration
        """
        average = self.elapsed_millis() // self._n_iters if self._n_iters > 0 else 0
        remaining = self.remaining_time_millis()
        return remaining > 2 * average and remaining > break_ms
    
    def end_iteration(self) -> None:
        """Mark the end of an iteration"""
        self._n_iters += 1
    
    def copy(self) -> 'ElapsedCpuTimer':
        """Create a copy of the timer"""
        new_timer = ElapsedCpuTimer()
        new_timer._max_time = self._max_time
        new_timer._old_time = self._old_time
        new_timer._n_iters = self._n_iters
        return new_timer
    
    def __str__(self) -> str:
        """String representation of elapsed time"""
        return f"{self.elapsed() / 1_000_000.0} ms elapsed"
    
    def _get_time(self) -> int:
        """Get current time in nanoseconds"""
        return self._get_cpu_time()
    
    def _get_cpu_time(self) -> int:
        """Get CPU time in nanoseconds"""
        if self._os_win:
            return time.time_ns()
        try:
            # On Unix systems, we could use process_time_ns() for CPU time
            return time.process_time_ns()
        except AttributeError:
            raise RuntimeError("CPU time not supported on this platform")