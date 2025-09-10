from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import threading
import multiprocessing
import time
import asyncio
import random
from enum import Enum
from datetime import datetime
from fastapi.staticfiles import StaticFiles

# فقط یک بار FastAPI app ایجاد کنید
app = FastAPI(title="Parallel Processing API", version="1.0.0")
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

# اضافه کردن این lines بعد از ایجاد app
@app.get("/")
async def read_root():
    return FileResponse("index.html")

@app.get("/style.css")
async def read_css():
    return FileResponse("style.css")

@app.get("/script.js")
async def read_js():
    return FileResponse("script.js")

# برای سرو کردن فایل‌های استاتیک
app.mount("/static", StaticFiles(directory="."), name="static")

# فقط یک بار middleware CORS اضافه کنید
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*","https://parallel-processing.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# بخش مشترک - تعاریف مدل ها
# ============================================================================

class SectionType(str, Enum):
    THREAD = "thread"
    PROCESS = "process"


class ScenarioRequest(BaseModel):
    section_type: SectionType
    section_name: str
    scenario_number: int
    parameters: Optional[Dict[str, Any]] = None


class ScenarioResponse(BaseModel):
    section_type: str
    section_name: str
    section_number: int
    scenario_number: int
    output: List[str]
    explanation: str


# ============================================================================
# بخش Thread
# ============================================================================

THREAD_SECTION_NAMES = {
    1: "Defining a thread",
    2: "Determining the current thread",
    3: "Thread subclass",
    4: "Synchronization with Lock",
    5: "Synchronization with RLock",
    6: "Synchronization with Semaphores",
    7: "Synchronization with Barrier"
}

THREAD_SECTION_REVERSE = {v: k for k, v in THREAD_SECTION_NAMES.items()}

# ============================================================================
# بخش Process
# ============================================================================

PROCESS_SECTION_NAMES = {
    1: "Spawning a process",
    2: "Naming a process",
    3: "Running processes in background",
    4: "Killing a process",
    5: "Process subclass",
    6: "Using queue for data exchange",
    7: "Synchronizing processes",
    8: "Using process pool"
}

PROCESS_SECTION_REVERSE = {v: k for k, v in PROCESS_SECTION_NAMES.items()}


# ============================================================================
# توابع global برای Process (باید در سطح ماژول باشند)
# ============================================================================

def myFunc_s1_sc1(i, lock, output):
    with lock:
        output.append(f"calling myFunc from process n°: {i}")
    time.sleep(0.1)
    for j in range(i):
        with lock:
            output.append(f"output from myFunc is :{j}")
        time.sleep(0.05)


def myFunc_s1_sc2(i, output):
    output.append(f"calling myFunc from process n°: {i}")
    for j in range(i):
        output.append(f"output from myFunc is :{j}")


def myFunc_s1_sc3(i, lock, output):
    delay = random.uniform(0.01, 0.1)
    time.sleep(delay)
    with lock:
        output.append(f"calling myFunc from process n°: {i}")
    for j in range(i):
        time.sleep(0.05)
        with lock:
            output.append(f"output from myFunc is :{j}")


def worker_s2_sc1(output):
    proc = multiprocessing.current_process()
    output.append(f"Starting process name = {proc.name}")
    time.sleep(1)
    output.append(f"Exiting process name = {proc.name}")


def task_s2_sc2(output):
    current = multiprocessing.current_process()
    output.append(f"Starting process name = {current.name}")
    time.sleep(0.5)
    output.append(f"Exiting process name = {current.name}")


def complex_task_s2_sc3(output):
    proc = multiprocessing.current_process()
    output.append(f"Starting process name = {proc.name}")
    if proc.name == "Process-2":
        proc.name = "Renamed-Process"
        output.append(f"Changed name to = {proc.name}")
    time.sleep(1)
    output.append(f"Exiting process name = {proc.name}")


def count_numbers_s3(process_name, start, end, output):
    output.append(f"Starting {process_name}")
    for i in range(start, end):
        output.append(f"---> {i}")
        time.sleep(0.1)
    output.append(f"Exiting {process_name}")


def long_running_task_s4_sc1(output):
    output.append("Task started")
    try:
        for i in range(10):
            output.append(f"Working... {i}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        output.append("Task interrupted")
    output.append("Task completed")


def infinite_task_s4_sc2(output):
    output.append("Infinite task started")
    try:
        while True:
            output.append("Still working...")
            time.sleep(0.3)
    except KeyboardInterrupt:
        output.append("Interrupted by signal")
    output.append("Task ended")


def graceful_task_s4_sc3(output):
    output.append("Graceful task started")
    for i in range(20):
        output.append(f"Processing item {i}")
        time.sleep(0.2)
        if multiprocessing.current_process().exitcode is not None:
            output.append("Received termination signal, cleaning up...")
            break
    output.append("Graceful task completed")


class MyProcess_s5_sc1(multiprocessing.Process):
    def __init__(self, process_id, output):
        super().__init__()
        self.process_id = process_id
        self.name = f"MyProcess-{process_id}"
        self.output = output

    def run(self):
        self.output.append(f"called run method by {self.name}")
        time.sleep(0.1)


class CustomProcess_s5_sc2(multiprocessing.Process):
    def __init__(self, process_id, delay, output):
        super().__init__()
        self.process_id = process_id
        self.delay = delay
        self.name = f"MyProcess-{process_id}"
        self.output = output

    def run(self):
        self.output.append(f"called run method by {self.name}")
        time.sleep(self.delay)
        self.output.append(f"{self.name} completed after {self.delay} seconds")


class AdvancedProcess_s5_sc3(multiprocessing.Process):
    def __init__(self, process_id, task_duration, output):
        super().__init__()
        self.process_id = process_id
        self.task_duration = task_duration
        self.name = f"MyProcess-{process_id}"
        self.stopped = multiprocessing.Event()
        self.output = output

    def run(self):
        self.output.append(f"called run method by {self.name}")
        start_time = time.time()
        while not self.stopped.is_set():
            time.sleep(self.task_duration)
            if time.time() - start_time > 1.0:
                break
        self.output.append(f"{self.name} finished execution")

    def stop(self):
        self.stopped.set()


def producer_s6_sc1(queue, producer_id, output):
    for i in range(5):
        item = random.randint(1, 100)
        queue.put(item)
        output.append(f"Process Producer : item {item} appended to queue {producer_id}")
        output.append(f"The size of queue is {queue.qsize()}")
        time.sleep(0.1)


def consumer_s6_sc1(queue, consumer_id, output):
    while not queue.empty():
        try:
            item = queue.get(timeout=1)
            output.append(f"Process Consumer : item {item} popped from by {consumer_id}")
            time.sleep(0.15)
        except:
            break


def worker_s6_sc2(task_queue, result_queue, worker_id, output):
    while True:
        try:
            task = task_queue.get(timeout=1)
            if task is None:
                break
            result = task * 2
            result_queue.put(result)
            output.append(f"Worker {worker_id} processed task {task} -> {result}")
            time.sleep(0.1)
        except:
            break


def priority_producer_s6_sc3(high_queue, low_queue, producer_id, output):
    for i in range(10):
        if random.random() > 0.7:
            high_queue.put(f"HIGH-{i}")
            output.append(f"Producer {producer_id} added HIGH priority item {i}")
        else:
            low_queue.put(f"LOW-{i}")
            output.append(f"Producer {producer_id} added LOW priority item {i}")
        time.sleep(0.1)


def priority_consumer_s6_sc3(high_queue, low_queue, consumer_id, output):
    count = 0
    while count < 10:
        if not high_queue.empty():
            item = high_queue.get()
            output.append(f"Consumer {consumer_id} processed {item}")
            count += 1
        elif not low_queue.empty():
            item = low_queue.get()
            output.append(f"Consumer {consumer_id} processed {item}")
            count += 1
        time.sleep(0.1)


def test_with_barrier_s7_sc1(barrier, process_name, output):
    time.sleep(0.1)
    barrier.wait()
    output.append(f"{process_name} - test_with_barrier ----> {datetime.now()}")


def test_without_barrier_s7_sc1(process_name, output):
    time.sleep(0.1)
    output.append(f"{process_name} - test_without_barrier ----> {datetime.now()}")


def shared_resource_access_s7_sc2(lock, process_name, output):
    time.sleep(0.1)
    with lock:
        current_time = datetime.now()
        output.append(f"{process_name} accessed shared resource at: {current_time}")
        time.sleep(0.05)


def independent_task_s7_sc2(process_name, output):
    time.sleep(0.1)
    output.append(f"{process_name} completed independent task at: {datetime.now()}")


def coordinated_task_s7_sc3(event, semaphore, process_name, output):
    event.wait()
    with semaphore:
        current_time = datetime.now()
        output.append(f"{process_name} started coordinated task at: {current_time}")
        time.sleep(random.uniform(0.1, 0.3))
        output.append(f"{process_name} finished coordinated task at: {datetime.now()}")


def independent_worker_s7_sc3(process_name, output):
    time.sleep(0.1)
    output.append(f"{process_name} working independently at: {datetime.now()}")


def square_s8_sc1(x):
    return x * x


def process_task_s8_sc2(x, output):
    time.sleep(0.01)
    result = x * x
    output.append(f"Processed {x} -> {result} by {multiprocessing.current_process().name}")
    return result


def complex_calculation_s8_sc3(x, output):
    time.sleep(0.02)
    result = x ** 2 + 2 * x + 1
    output.append(f"Calculated f({x}) = {result}")
    return result


# ============================================================================
# توابع اجرای سناریوهای Thread
# ============================================================================

async def run_thread_section1(scenario_number: int, parameters: Optional[Dict] = None):
    """بخش 1: Defining a thread"""
    output_lines = []
    output_lock = threading.Lock()

    if scenario_number == 1:
        def worker(thread_id, lock, output):
            with lock:
                output.append(f"my_func called by thread N°{thread_id}")

        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i, output_lock, output_lines))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        explanation = "ده thread به صورت موازی ایجاد و اجرا می‌شوند"

    elif scenario_number == 2:
        def worker(thread_id, delay, lock, output):
            time.sleep(delay)
            with lock:
                output.append(f"Thread {thread_id} finished after {delay:.1f} seconds")

        threads = []
        delays = [0.1, 0.3, 0.2, 0.5, 0.4, 0.7, 0.6, 0.9, 0.8, 1.0]
        for i in range(10):
            t = threading.Thread(target=worker, args=(i, delays[i], output_lock, output_lines))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        explanation = "ده thread با تاخیرهای مختلف ایجاد می‌شوند"

    elif scenario_number == 3:
        def worker(lock, output):
            thread_name = threading.current_thread().name
            with lock:
                output.append(f"{thread_name} is running")

        threads = []
        names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa"]
        for name in names:
            t = threading.Thread(target=worker, name=name, args=(output_lock, output_lines))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        explanation = "ده thread با نام‌های یونانی ایجاد می‌شوند"

    else:
        raise HTTPException(status_code=400, detail="سناریو thread یافت نشد")

    return {"output": output_lines, "explanation": explanation}


async def run_thread_section2(scenario_number: int, parameters: Optional[Dict] = None):
    """بخش 2: Determining the current thread"""
    output_lines = []
    output_lock = threading.Lock()

    if scenario_number == 1:
        def function_A(lock, output):
            with lock:
                output.append("function_A-> starting")
            time.sleep(1)
            with lock:
                output.append("function_A-> exiting")

        def function_B(lock, output):
            with lock:
                output.append("function_B-> starting")
            time.sleep(0.5)
            with lock:
                output.append("function_B-> exiting")

        def function_C(lock, output):
            with lock:
                output.append("function_C-> starting")
            time.sleep(0.8)
            with lock:
                output.append("function_C-> exiting")

        thread_A = threading.Thread(target=function_A, args=(output_lock, output_lines))
        thread_B = threading.Thread(target=function_B, args=(output_lock, output_lines))
        thread_C = threading.Thread(target=function_C, args=(output_lock, output_lines))

        thread_A.start()
        thread_B.start()
        thread_C.start()

        thread_A.join()
        thread_B.join()
        thread_C.join()

        explanation = "سه تابع مختلف در threadهای جداگانه اجرا می‌شوند"

    elif scenario_number == 2:
        def worker(thread_id, lock, output):
            current_thread = threading.current_thread()
            with lock:
                output.append(f"Thread {thread_id}: Name={current_thread.name}, ID={current_thread.ident}")
            time.sleep(0.5)

        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i, output_lock, output_lines), name=f"Worker-{i}")
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        explanation = "نمایش اطلاعات thread جاری شامل نام و شناسه"

    elif scenario_number == 3:
        def daemon_worker(lock, output):
            with lock:
                output.append("Daemon thread started")
            time.sleep(2)
            with lock:
                output.append("This won't be printed!")

        def normal_worker(lock, output):
            with lock:
                output.append("Normal thread started")
            time.sleep(1)
            with lock:
                output.append("Normal thread exiting")

        daemon_thread = threading.Thread(target=daemon_worker, args=(output_lock, output_lines), daemon=True)
        normal_thread = threading.Thread(target=normal_worker, args=(output_lock, output_lines))

        daemon_thread.start()
        normal_thread.start()

        normal_thread.join()
        with output_lock:
            output_lines.append("Main thread exiting")

        explanation = "مقایسه رفتار threadهای daemon و normal"

    else:
        raise HTTPException(status_code=400, detail="سناریو thread یافت نشد")

    return {"output": output_lines, "explanation": explanation}


async def run_thread_section3(scenario_number: int, parameters: Optional[Dict] = None):
    """بخش 3: Thread subclass"""
    output_lines = []
    output_lock = threading.Lock()

    if scenario_number == 1:
        class CustomThread(threading.Thread):
            def __init__(self, thread_id, lock, output):
                super().__init__()
                self.thread_id = thread_id
                self.lock = lock
                self.output = output

            def run(self):
                with self.lock:
                    self.output.append(
                        f"--> Thread#{self.thread_id} running, belonging to process ID {threading.get_ident()}")

        threads = []
        for i in range(1, 10):
            t = CustomThread(i, output_lock, output_lines)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        explanation = "ایجاد thread با subclass و نمایش اطلاعات process"

    elif scenario_number == 2:
        class CountingThread(threading.Thread):
            def __init__(self, name, count, lock, output):
                super().__init__()
                self.name = name
                self.count = count
                self.lock = lock
                self.output = output

            def run(self):
                for i in range(self.count):
                    with self.lock:
                        self.output.append(f"{self.name}: {i}")
                    time.sleep(0.1)

        thread1 = CountingThread("Counter-A", 5, output_lock, output_lines)
        thread2 = CountingThread("Counter-B", 3, output_lock, output_lines)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        explanation = "Threadهای شمارنده با پارامترهای مختلف"

    elif scenario_number == 3:
        class TimedThread(threading.Thread):
            def __init__(self, duration, lock, output):
                super().__init__()
                self.duration = duration
                self.lock = lock
                self.output = output
                self.result = None

            def run(self):
                start_time = time.time()
                time.sleep(self.duration)
                self.result = time.time() - start_time
                with self.lock:
                    self.output.append(f"Thread completed in {self.result:.2f} seconds")

        threads = []
        durations = [1.0, 2.0, 0.5, 1.5]
        for duration in durations:
            t = TimedThread(duration, output_lock, output_lines)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        explanation = "Threadهای زمان‌بندی شده با مدت زمان مختلف"

    else:
        raise HTTPException(status_code=400, detail="سناریو thread یافت نشد")

    return {"output": output_lines, "explanation": explanation}


async def run_thread_section4(scenario_number: int, parameters: Optional[Dict] = None):
    """بخش 4: Synchronization with Lock"""
    output_lines = []
    output_lock = threading.Lock()

    if scenario_number == 1:
        lock = threading.Lock()

        def task(thread_id, lock, output_lock, output):
            with lock:
                with output_lock:
                    output.append(f"--> Thread#{thread_id} running, belonging to process ID {threading.get_ident()}")
                time.sleep(0.5)
                with output_lock:
                    output.append(f"--> Thread#{thread_id} over")

        threads = []
        for i in range(1, 10):
            t = threading.Thread(target=task, args=(i, lock, output_lock, output_lines))
            threads.append(t)
            t.start()
            time.sleep(0.1)

        for t in threads:
            t.join()

        with output_lock:
            output_lines.append("End")
        explanation = "استفاده از Lock برای همگام‌سازی"

    elif scenario_number == 2:
        lock = threading.Lock()
        shared_list = []

        def add_to_list(thread_id, value, lock, output_lock, output):
            with lock:
                with output_lock:
                    output.append(f"Thread {thread_id} acquired lock")
                shared_list.append(value)
                time.sleep(0.2)
                with output_lock:
                    output.append(f"Thread {thread_id} added {value}, list: {shared_list}")
                    output.append(f"Thread {thread_id} released lock")

        threads = []
        for i in range(5):
            t = threading.Thread(target=add_to_list, args=(i, i * 10, lock, output_lock, output_lines))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        explanation = "همگام‌سازی دسترسی به لیست مشترک با Lock"

    elif scenario_number == 3:
        lock = threading.Lock()

        def task(thread_id, delay, lock, output_lock, output):
            with output_lock:
                output.append(f"Thread {thread_id} waiting for lock...")
            with lock:
                with output_lock:
                    output.append(f"Thread {thread_id} acquired lock")
                time.sleep(delay)
                with output_lock:
                    output.append(f"Thread {thread_id} releasing lock after {delay} seconds")

        threads = []
        delays = [1.0, 0.5, 0.8, 0.3, 0.6]
        for i, delay in enumerate(delays):
            t = threading.Thread(target=task, args=(i, delay, lock, output_lock, output_lines))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        explanation = "نمایش رفتار Lock با تاخیرهای مختلف"

    else:
        raise HTTPException(status_code=400, detail="سناریو thread یافت نشد")

    return {"output": output_lines, "explanation": explanation}


async def run_thread_section5(scenario_number: int, parameters: Optional[Dict] = None):
    """بخش 5: Synchronization with RLock"""
    output_lines = []
    output_lock = threading.Lock()

    if scenario_number == 1:
        rlock = threading.RLock()
        items_to_add = 16
        items_to_remove = 1

        def add_items(rlock, output_lock, output):
            nonlocal items_to_add
            with rlock:
                while items_to_add > 0:
                    items_to_add -= 1
                    with output_lock:
                        output.append(f"ADDED one item →{items_to_add} item to ADD")
                    time.sleep(0.1)

        def remove_items(rlock, output_lock, output):
            nonlocal items_to_remove
            with rlock:
                while items_to_remove > 0:
                    items_to_remove -= 1
                    with output_lock:
                        output.append(f"REMOVED one item →{items_to_remove} item to REMOVE")
                    time.sleep(0.1)

        with output_lock:
            output_lines.append(f"N° {16} items to ADD")
            output_lines.append(f"N° {1} items to REMOVE")

        add_thread = threading.Thread(target=add_items, args=(rlock, output_lock, output_lines))
        remove_thread = threading.Thread(target=remove_items, args=(rlock, output_lock, output_lines))

        add_thread.start()
        remove_thread.start()

        add_thread.join()
        remove_thread.join()

        explanation = "استفاده از RLock برای عملیات همزمان"

    elif scenario_number == 2:
        class SharedResource:
            def __init__(self, output_lock, output):
                self.rlock = threading.RLock()
                self.data = []
                self.output_lock = output_lock
                self.output = output

            def add_data(self, value):
                with self.rlock:
                    self.data.append(value)
                    with self.output_lock:
                        self.output.append(f"Added {value}, data: {self.data}")

            def process_data(self):
                with self.rlock:
                    if self.data:
                        value = self.data.pop(0)
                        with self.output_lock:
                            self.output.append(f"Processed {value}, remaining: {self.data}")

        resource = SharedResource(output_lock, output_lines)

        def producer():
            for i in range(5):
                resource.add_data(i)
                time.sleep(0.2)

        def consumer():
            for i in range(5):
                resource.process_data()
                time.sleep(0.3)

        prod_thread = threading.Thread(target=producer)
        cons_thread = threading.Thread(target=consumer)

        prod_thread.start()
        cons_thread.start()

        prod_thread.join()
        cons_thread.join()

        explanation = "الگوی Producer-Consumer با RLock"

    elif scenario_number == 3:
        rlock = threading.RLock()

        def recursive_function(level, rlock, output_lock, output):
            with rlock:
                if level > 0:
                    with output_lock:
                        output.append(f"Level {level}: Acquired lock")
                    recursive_function(level - 1, rlock, output_lock, output)
                    with output_lock:
                        output.append(f"Level {level}: Releasing lock")

        threading.Thread(target=recursive_function, args=(3, rlock, output_lock, output_lines)).start()
        time.sleep(1)

        explanation = "نمایش قابلیت بازگشتی RLock"

    else:
        raise HTTPException(status_code=400, detail="سناریو thread یافت نشد")

    return {"output": output_lines, "explanation": explanation}


async def run_thread_section6(scenario_number: int, parameters: Optional[Dict] = None):
    """بخش 6: Synchronization with Semaphores"""
    output_lines = []
    output_lock = threading.Lock()

    if scenario_number == 1:
        buffer = []
        MAX_ITEMS = 5
        empty = threading.Semaphore(MAX_ITEMS)
        full = threading.Semaphore(0)
        mutex = threading.Lock()

        def producer(producer_id, empty, full, mutex, output_lock, output):
            for i in range(3):
                item = random.randint(1, 1000)
                empty.acquire()
                with mutex:
                    buffer.append(item)
                    with output_lock:
                        output.append(f"Producer notify: item number {item}")
                full.release()
                time.sleep(random.uniform(2, 4))

        def consumer(consumer_id, empty, full, mutex, output_lock, output):
            for i in range(3):
                full.acquire()
                with mutex:
                    if buffer:
                        item = buffer.pop(0)
                        with output_lock:
                            output.append(f"Consumer notify: item number {item}")
                empty.release()
                time.sleep(random.uniform(1, 3))

        producers = []
        consumers = []

        for i in range(1, 21, 2):
            p = threading.Thread(target=producer, args=(i, empty, full, mutex, output_lock, output_lines))
            producers.append(p)

        for i in range(2, 21, 2):
            c = threading.Thread(target=consumer, args=(i, empty, full, mutex, output_lock, output_lines))
            consumers.append(c)

        for c in consumers:
            c.start()
            time.sleep(0.1)

        for p in producers:
            p.start()
            time.sleep(0.1)

        for p in producers:
            p.join()

        for c in consumers:
            c.join()

        explanation = "الگوی Producer-Consumer با Semaphore"

    elif scenario_number == 2:
        max_connections = 3
        connection_semaphore = threading.Semaphore(max_connections)

        def database_connection(connection_id, semaphore, output_lock, output):
            with semaphore:
                with output_lock:
                    output.append(f"Connection {connection_id}: Established")
                time.sleep(1)
                with output_lock:
                    output.append(f"Connection {connection_id}: Closed")

        threads = []
        for i in range(8):
            t = threading.Thread(target=database_connection, args=(i, connection_semaphore, output_lock, output_lines))
            threads.append(t)
            t.start()
            time.sleep(0.2)

        for t in threads:
            t.join()

        explanation = "محدود کردن تعداد اتصالات همزمان با Semaphore"

    elif scenario_number == 3:
        ticket_semaphore = threading.Semaphore(5)

        def buy_ticket(customer_id, semaphore, output_lock, output):
            if semaphore.acquire(blocking=False):
                with output_lock:
                    output.append(f"Customer {customer_id}: Ticket purchased")
                time.sleep(0.5)
                semaphore.release()
                with output_lock:
                    output.append(f"Customer {customer_id}: Done")
            else:
                with output_lock:
                    output.append(f"Customer {customer_id}: No tickets available")

        threads = []
        for i in range(10):
            t = threading.Thread(target=buy_ticket, args=(i, ticket_semaphore, output_lock, output_lines))
            threads.append(t)
            t.start()
            time.sleep(0.1)

        for t in threads:
            t.join()

        explanation = "مدیریت منابع محدود با Semaphore"

    else:
        raise HTTPException(status_code=400, detail="سناریو thread یافت نشد")

    return {"output": output_lines, "explanation": explanation}


async def run_thread_section7(scenario_number: int, parameters: Optional[Dict] = None):
    """بخش 7: Synchronization with Barrier"""
    output_lines = []
    output_lock = threading.Lock()

    if scenario_number == 1:
        num_racers = 3
        barrier = threading.Barrier(num_racers, action=lambda: output_lines.append("START RACE!!!!"))

        def racer(name, barrier, output_lock, output):
            with output_lock:
                output.append(f"{name} reached the barrier at: {time.ctime()}")
            barrier.wait()
            time.sleep(random.uniform(1.0, 3.0))
            with output_lock:
                output.append(f"{name} finished!")

        threads = []
        racers = ["Dewey", "Huey", "Louie"]

        for name in racers:
            t = threading.Thread(target=racer, args=(name, barrier, output_lock, output_lines))
            threads.append(t)
            t.start()
            time.sleep(1)

        for t in threads:
            t.join()

        with output_lock:
            output_lines.append("Race over!")
        explanation = "مسابقه با Barrier برای هماهنگی شروع"

    elif scenario_number == 2:
        num_workers = 4
        barrier = threading.Barrier(num_workers)

        def worker(worker_id, barrier, output_lock, output):
            with output_lock:
                output.append(f"Worker {worker_id}: Phase 1 started")
            time.sleep(random.uniform(0.5, 2.0))
            with output_lock:
                output.append(f"Worker {worker_id}: Waiting at barrier")
            barrier.wait()
            with output_lock:
                output.append(f"Worker {worker_id}: Phase 2 started")
            time.sleep(0.5)
            with output_lock:
                output.append(f"Worker {worker_id}: Completed")

        threads = []
        for i in range(num_workers):
            t = threading.Thread(target=worker, args=(i, barrier, output_lock, output_lines))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        explanation = "هماهنگی فازهای کاری با Barrier"

    elif scenario_number == 3:
        num_teams = 2
        players_per_team = 3
        barrier = threading.Barrier(players_per_team * num_teams)

        def player(team_id, player_id, barrier, output_lock, output):
            with output_lock:
                output.append(f"Team {team_id} Player {player_id}: Ready")
            barrier.wait()
            with output_lock:
                output.append(f"Team {team_id} Player {player_id}: Game started!")

        threads = []
        for team in range(num_teams):
            for player_num in range(players_per_team):
                t = threading.Thread(target=player, args=(team, player_num, barrier, output_lock, output_lines))
                threads.append(t)
                t.start()
                time.sleep(0.3)

        for t in threads:
            t.join()

        with output_lock:
            output_lines.append("All players started successfully!")
        explanation = "هماهنگی بازیکنان تیم‌های مختلف با Barrier"

    else:
        raise HTTPException(status_code=400, detail="سناریو thread یافت نشد")

    return {"output": output_lines, "explanation": explanation}


# ============================================================================
# توابع اجرای سناریوهای Process
# ============================================================================

async def run_process_section1(scenario_number: int, parameters: Optional[Dict] = None):
    """بخش 1: Spawning a process"""
    output_lines = []
    manager = multiprocessing.Manager()
    shared_output = manager.list()
    output_lock = multiprocessing.Lock()

    if scenario_number == 1:
        processes = []
        for i in range(6):
            p = multiprocessing.Process(target=myFunc_s1_sc1, args=(i, output_lock, shared_output))
            processes.append(p)
            p.start()
            time.sleep(0.05)

        for p in processes:
            p.join()

        output_lines.extend(shared_output)
        explanation = "ایجاد و اجرای چندین فرآیند به صورت موازی"

    elif scenario_number == 2:
        processes = []
        for i in range(6):
            p = multiprocessing.Process(target=myFunc_s1_sc2, args=(i, shared_output))
            processes.append(p)
            p.start()
            p.join()

        output_lines.extend(shared_output)
        explanation = "ایجاد فرآیندها به صورت سریال با join کردن هر کدام"

    elif scenario_number == 3:
        processes = []
        for i in range(6):
            p = multiprocessing.Process(target=myFunc_s1_sc3, args=(i, output_lock, shared_output))
            processes.append(p)
            p.start()
            time.sleep(0.01)

        for p in processes:
            p.join()

        output_lines.extend(shared_output)
        explanation = "ایجاد فرآیندها با تاخیرهای تصادفی و استفاده از lock برای هماهنگی"

    else:
        raise HTTPException(status_code=400, detail="سناریو process یافت نشد")

    return {"output": output_lines, "explanation": explanation}


async def run_process_section2(scenario_number: int, parameters: Optional[Dict] = None):
    """بخش 2: Naming a process"""
    output_lines = []
    manager = multiprocessing.Manager()
    shared_output = manager.list()

    if scenario_number == 1:
        p1 = multiprocessing.Process(target=worker_s2_sc1, name="myFunc process", args=(shared_output,))
        p2 = multiprocessing.Process(target=worker_s2_sc1, args=(shared_output,))

        p1.start()
        p2.start()

        p1.join()
        p2.join()

        output_lines.extend(shared_output)
        explanation = "ایجاد فرآیند با نام دستی و نام پیش‌فرض"

    elif scenario_number == 2:
        processes = []
        p1 = multiprocessing.Process(target=task_s2_sc2, name="myFunc process", args=(shared_output,))
        processes.append(p1)

        for i in range(2):
            p = multiprocessing.Process(target=task_s2_sc2, args=(shared_output,))
            processes.append(p)

        for p in processes:
            p.start()
            time.sleep(0.1)

        for p in processes:
            p.join()

        output_lines.extend(shared_output)
        explanation = "ایجاد چندین فرآیند با نام‌های مختلف"

    elif scenario_number == 3:
        p1 = multiprocessing.Process(target=complex_task_s2_sc3, name="myFunc process", args=(shared_output,))
        p2 = multiprocessing.Process(target=complex_task_s2_sc3, args=(shared_output,))

        p1.start()
        p2.start()

        p1.join()
        p2.join()

        output_lines.extend(shared_output)
        explanation = "تغییر نام فرآیند در حین اجرا"

    else:
        raise HTTPException(status_code=400, detail="سناریو process یافت نشد")

    return {"output": output_lines, "explanation": explanation}


async def run_process_section3(scenario_number: int, parameters: Optional[Dict] = None):
    """بخش 3: Running processes in background"""
    output_lines = []
    manager = multiprocessing.Manager()
    shared_output = manager.list()

    if scenario_number == 1:
        p1 = multiprocessing.Process(target=count_numbers_s3,
                                     args=("NO_background_process", 5, 10, shared_output))
        p1.start()
        p1.join()

        output_lines.extend(shared_output)
        output_lines.append("Main process continues after normal process completion")
        explanation = "فرآیند عادی با انتظار برای پایان آن"

    elif scenario_number == 2:
        p1 = multiprocessing.Process(target=count_numbers_s3,
                                     args=("NO_background_process", 5, 10, shared_output))
        p2 = multiprocessing.Process(target=count_numbers_s3,
                                     args=("background_process", 0, 5, shared_output))

        p1.start()
        p2.start()
        p1.join()

        output_lines.extend(shared_output)
        output_lines.append("Main process continues without waiting for background process")
        time.sleep(1)
        explanation = "فرآیند پس‌زمینه بدون انتظار برای پایان آن"

    elif scenario_number == 3:
        p1 = multiprocessing.Process(target=count_numbers_s3,
                                     args=("NO_background_process", 5, 10, shared_output))
        p2 = multiprocessing.Process(target=count_numbers_s3,
                                     args=("background_process", 0, 5, shared_output))
        p2.daemon = True

        p1.start()
        p2.start()
        p1.join()

        output_lines.extend(shared_output)
        output_lines.append("Main process exits, daemon process will be terminated automatically")
        explanation = "فرآیند دیمون که با پایان فرآیند اصلی terminate می‌شود"

    else:
        raise HTTPException(status_code=400, detail="سناریو process یافت نشد")

    return {"output": output_lines, "explanation": explanation}


async def run_process_section4(scenario_number: int, parameters: Optional[Dict] = None):
    """بخش 4: Killing a process"""
    output_lines = []
    manager = multiprocessing.Manager()
    shared_output = manager.list()

    if scenario_number == 1:
        process = multiprocessing.Process(target=long_running_task_s4_sc1, args=(shared_output,))
        output_lines.append(f"Process before execution: {process} {process.is_alive()}")

        process.start()
        time.sleep(0.1)
        output_lines.append(f"Process running: {process} {process.is_alive()}")

        time.sleep(1)
        process.terminate()
        output_lines.append(f"Process terminated: {process} {process.is_alive()}")

        process.join()
        output_lines.append(f"Process joined: {process} {process.is_alive()}")
        output_lines.append(f"Process exit code: {process.exitcode}")

        output_lines.extend(shared_output)
        explanation = "پایان دادن به فرآیند با terminate()"

    elif scenario_number == 2:
        process = multiprocessing.Process(target=infinite_task_s4_sc2, args=(shared_output,))
        output_lines.append(f"Process before execution: {process} {process.is_alive()}")

        process.start()
        time.sleep(0.1)
        output_lines.append(f"Process running: {process} {process.is_alive()}")

        time.sleep(1)
        process.kill()
        output_lines.append(f"Process killed: {process} {process.is_alive()}")

        process.join()
        output_lines.append(f"Process joined: {process} {process.is_alive()}")
        output_lines.append(f"Process exit code: {process.exitcode}")

        output_lines.extend(shared_output)
        explanation = "پایان شدید فرآیند با kill() (معادل SIGKILL)"

    elif scenario_number == 3:
        process = multiprocessing.Process(target=graceful_task_s4_sc3, args=(shared_output,))
        output_lines.append(f"Process before execution: {process} {process.is_alive()}")

        process.start()
        time.sleep(0.1)
        output_lines.append(f"Process running: {process} {process.is_alive()}")

        time.sleep(1.5)
        process.terminate()
        output_lines.append(f"Process terminated: {process} {process.is_alive()}")

        process.join()
        output_lines.append(f"Process joined: {process} {process.is_alive()}")
        output_lines.append(f"Process exit code: {process.exitcode}")

        output_lines.extend(shared_output)
        explanation = "پایان graceful فرآیند با بررسی exitcode"

    else:
        raise HTTPException(status_code=400, detail="سناریو process یافت نشد")

    return {"output": output_lines, "explanation": explanation}


async def run_process_section5(scenario_number: int, parameters: Optional[Dict] = None):
    """بخش 5: Process subclass"""
    output_lines = []
    manager = multiprocessing.Manager()
    shared_output = manager.list()

    if scenario_number == 1:
        processes = []
        for i in range(1, 11):
            p = MyProcess_s5_sc1(i, shared_output)
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        output_lines.extend(shared_output)
        explanation = "ایجاد فرآیند با subclass پایه"

    elif scenario_number == 2:
        processes = []
        for i in range(1, 11):
            delay = i * 0.05
            p = CustomProcess_s5_sc2(i, delay, shared_output)
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        output_lines.extend(shared_output)
        explanation = "ایجاد فرآیند با تاخیرهای قابل تنظیم"

    elif scenario_number == 3:
        processes = []
        for i in range(1, 11):
            duration = random.uniform(0.1, 0.3)
            p = AdvancedProcess_s5_sc3(i, duration, shared_output)
            processes.append(p)
            p.start()

        time.sleep(0.5)
        for p in processes:
            p.stop()

        for p in processes:
            p.join()

        output_lines.extend(shared_output)
        explanation = "ایجاد فرآیند پیشرفته با قابلیت توقف"

    else:
        raise HTTPException(status_code=400, detail="سناریو process یافت نشد")

    return {"output": output_lines, "explanation": explanation}


async def run_process_section6(scenario_number: int, parameters: Optional[Dict] = None):
    """بخش 6: Using queue for data exchange"""
    output_lines = []
    manager = multiprocessing.Manager()
    shared_output = manager.list()

    if scenario_number == 1:
        queue = multiprocessing.Queue()
        p1 = multiprocessing.Process(target=producer_s6_sc1, args=(queue, "producer-1", shared_output))
        p2 = multiprocessing.Process(target=consumer_s6_sc1, args=(queue, "consumer-2", shared_output))

        p1.start()
        p2.start()

        p1.join()
        p2.join()

        output_lines.extend(shared_output)
        output_lines.append("the queue is empty")
        explanation = "مبادله داده بین فرآیندها با استفاده از Queue"

    elif scenario_number == 2:
        task_queue = multiprocessing.Queue()
        result_queue = multiprocessing.Queue()

        for i in range(10):
            task_queue.put(i)

        workers = []
        for i in range(3):
            p = multiprocessing.Process(target=worker_s6_sc2, args=(task_queue, result_queue, i + 1, shared_output))
            workers.append(p)
            p.start()

        for p in workers:
            task_queue.put(None)

        for p in workers:
            p.join()

        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        output_lines.extend(shared_output)
        output_lines.append(f"Results: {results}")
        explanation = "الگوی Worker با استفاده از Queue"

    elif scenario_number == 3:
        high_queue = multiprocessing.Queue()
        low_queue = multiprocessing.Queue()

        p1 = multiprocessing.Process(target=priority_producer_s6_sc3, args=(high_queue, low_queue, "P1", shared_output))
        p2 = multiprocessing.Process(target=priority_consumer_s6_sc3, args=(high_queue, low_queue, "C1", shared_output))

        p1.start()
        p2.start()

        p1.join()
        p2.join()

        output_lines.extend(shared_output)
        explanation = "Queue با اولویت‌بندی"

    else:
        raise HTTPException(status_code=400, detail="سناریو process یافت نشد")

    return {"output": output_lines, "explanation": explanation}


async def run_process_section7(scenario_number: int, parameters: Optional[Dict] = None):
    """بخش 7: Synchronizing processes"""
    output_lines = []
    manager = multiprocessing.Manager()
    shared_output = manager.list()

    if scenario_number == 1:
        barrier = multiprocessing.Barrier(2)
        p1 = multiprocessing.Process(target=test_with_barrier_s7_sc1, args=(barrier, "process p1", shared_output))
        p2 = multiprocessing.Process(target=test_with_barrier_s7_sc1, args=(barrier, "process p2", shared_output))
        p3 = multiprocessing.Process(target=test_without_barrier_s7_sc1, args=("process p3", shared_output))
        p4 = multiprocessing.Process(target=test_without_barrier_s7_sc1, args=("process p4", shared_output))

        p1.start()
        p2.start()
        p3.start()
        p4.start()

        p1.join()
        p2.join()
        p3.join()
        p4.join()

        output_lines.extend(shared_output)
        explanation = "هماهنگی فرآیندها با Barrier"

    elif scenario_number == 2:
        lock = multiprocessing.Lock()
        processes = []

        for i in range(2):
            p = multiprocessing.Process(target=shared_resource_access_s7_sc2,
                                        args=(lock, f"process p{i + 1}", shared_output))
            processes.append(p)
            p.start()

        for i in range(2):
            p = multiprocessing.Process(target=independent_task_s7_sc2, args=(f"process p{i + 3}", shared_output))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        output_lines.extend(shared_output)
        explanation = "هماهنگی با Lock برای دسترسی انحصاری"

    elif scenario_number == 3:
        start_event = multiprocessing.Event()
        semaphore = multiprocessing.Semaphore(2)
        processes = []

        for i in range(4):
            p = multiprocessing.Process(target=coordinated_task_s7_sc3,
                                        args=(start_event, semaphore, f"coordinated p{i + 1}", shared_output))
            processes.append(p)
            p.start()

        for i in range(2):
            p = multiprocessing.Process(target=independent_worker_s7_sc3, args=(f"independent p{i + 1}", shared_output))
            processes.append(p)
            p.start()

        time.sleep(0.5)
        output_lines.append("Starting coordinated tasks...")
        start_event.set()

        for p in processes:
            p.join()

        output_lines.extend(shared_output)
        explanation = "هماهنگی پیشرفته با Event و Semaphore"

    else:
        raise HTTPException(status_code=400, detail="سناریو process یافت نشد")

    return {"output": output_lines, "explanation": explanation}


async def run_process_section8(scenario_number: int, parameters: Optional[Dict] = None):
    """بخش 8: Using process pool"""
    output_lines = []
    manager = multiprocessing.Manager()
    shared_output = manager.list()

    if scenario_number == 1:
        with multiprocessing.Pool(processes=4) as pool:
            results = pool.map(square_s8_sc1, range(100))
            output_lines.append(f"Pool : {results}")

        explanation = "استفاده از Process Pool برای محاسبه مربع اعداد"

    elif scenario_number == 2:
        def process_task_wrapper(x):
            return process_task_s8_sc2(x, shared_output)

        num_processes = multiprocessing.cpu_count()
        output_lines.append(f"Using {num_processes} processes")

        with multiprocessing.Pool(processes=num_processes) as pool:
            async_result = pool.map_async(process_task_wrapper, range(100))
            output_lines.append("Main process is doing other work...")
            time.sleep(0.5)
            results = async_result.get()

            output_lines.append(f"Total results: {len(results)}")
            output_lines.append(f"First 10 results: {results[:10]}")
            output_lines.append(f"Last 10 results: {results[-10:]}")

        explanation = "استفاده از map_async برای اجرای غیرمسدود کننده"

    elif scenario_number == 3:
        def complex_calculation_wrapper(x):
            return complex_calculation_s8_sc3(x, shared_output)

        with multiprocessing.Pool(processes=3) as pool:
            results = list(pool.imap(complex_calculation_wrapper, range(20)))
            output_lines.append(f"All results: {results}")

        explanation = "استفاده از imap برای نتایج تدریجی"

    else:
        raise HTTPException(status_code=400, detail="سناریو process یافت نشد")

    return {"output": output_lines, "explanation": explanation}


# ============================================================================
# endpoint اصلی
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Parallel Processing API is running",
        "thread_sections": THREAD_SECTION_NAMES,
        "process_sections": PROCESS_SECTION_NAMES
    }


@app.post("/api/run-scenario", response_model=ScenarioResponse)
async def run_scenario(request: ScenarioRequest):
    try:
        if request.section_type == SectionType.THREAD:
            if request.section_name not in THREAD_SECTION_REVERSE:
                raise HTTPException(status_code=400, detail="بخش thread یافت نشد")

            section_number = THREAD_SECTION_REVERSE[request.section_name]

            if section_number == 1:
                result = await run_thread_section1(request.scenario_number, request.parameters)
            elif section_number == 2:
                result = await run_thread_section2(request.scenario_number, request.parameters)
            elif section_number == 3:
                result = await run_thread_section3(request.scenario_number, request.parameters)
            elif section_number == 4:
                result = await run_thread_section4(request.scenario_number, request.parameters)
            elif section_number == 5:
                result = await run_thread_section5(request.scenario_number, request.parameters)
            elif section_number == 6:
                result = await run_thread_section6(request.scenario_number, request.parameters)
            elif section_number == 7:
                result = await run_thread_section7(request.scenario_number, request.parameters)
            else:
                raise HTTPException(status_code=400, detail="بخش thread یافت نشد")

        elif request.section_type == SectionType.PROCESS:
            if request.section_name not in PROCESS_SECTION_REVERSE:
                raise HTTPException(status_code=400, detail="بخش process یافت نشد")

            section_number = PROCESS_SECTION_REVERSE[request.section_name]

            if section_number == 1:
                result = await run_process_section1(request.scenario_number, request.parameters)
            elif section_number == 2:
                result = await run_process_section2(request.scenario_number, request.parameters)
            elif section_number == 3:
                result = await run_process_section3(request.scenario_number, request.parameters)
            elif section_number == 4:
                result = await run_process_section4(request.scenario_number, request.parameters)
            elif section_number == 5:
                result = await run_process_section5(request.scenario_number, request.parameters)
            elif section_number == 6:
                result = await run_process_section6(request.scenario_number, request.parameters)
            elif section_number == 7:
                result = await run_process_section7(request.scenario_number, request.parameters)
            elif section_number == 8:
                result = await run_process_section8(request.scenario_number, request.parameters)
            else:
                raise HTTPException(status_code=400, detail="بخش process یافت نشد")
        else:
            raise HTTPException(status_code=400, detail="نوع section نامعتبر است")

        return ScenarioResponse(
            section_type=request.section_type.value,
            section_name=request.section_name,
            section_number=section_number,
            scenario_number=request.scenario_number,
            output=result["output"],
            explanation=result["explanation"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطا در اجرای سناریو: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)