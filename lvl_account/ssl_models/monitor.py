import psutil
import time
import threading

def monitor_resources(stop_event, interval=1.0):
    """Monitor and print CPU and disk I/O stats at intervals"""
    process = psutil.Process()
    
    print("Time | CPU% | RSS MB | VMS MB | Read MB | Write MB")
    print("-" * 60)
    div = 1024**2
    
    start_time = time.time()
    io_counters = psutil.disk_io_counters()
    last_read = io_counters.read_bytes / div
    last_write = io_counters.write_bytes / div
    
    while not stop_event.is_set():
        with process.oneshot():
            # Get CPU usage
            cpu_percent = process.cpu_percent()
            
            # Get memory usage
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / div  # Resident Set Size (physical memory)
            vms_mb = mem_info.vms / div  # Virtual Memory Size
            
            # Get I/O counters
            io_counters = psutil.disk_io_counters()
            read_mb = io_counters.read_bytes / div
            write_mb = io_counters.write_bytes / div
            
            read_rate = read_mb - last_read
            write_rate = write_mb - last_write
            
            elapsed = time.time() - start_time
            print(f"{elapsed:.1f}s | {cpu_percent:5.1f}% | {rss_mb:6.1f} | {vms_mb:6.1f} | {read_rate:6.1f} | {write_rate:6.1f}")
            
            last_read, last_write = read_mb, write_mb
            time.sleep(interval)

class Monitor:
    def __init__(self):
        # Create an event to signal monitor thread to stop
        self.stop_event = threading.Event()
        self.monitor_thread = threading.Thread(target=monitor_resources, args=(self.stop_event,))
        self.monitor_thread.daemon = True

    def __enter__(self):
        # Start monitoring in background thread
        self.monitor_thread.start()
        return self
    

    def __exit__(self, exc_type, exc_value, traceback):
        # Stop the monitoring thread
        self.stop_event.set()
        self.monitor_thread.join(timeout=1.0)