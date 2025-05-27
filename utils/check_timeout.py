import multiprocessing
import time

def check_timeout_column_name(func, timeout):
    def wrapper(queue):
        result = func()
        queue.put(result)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapper, args=(queue,))
    process.start()
    process.join(timeout)  # 타임아웃 시간 동안 실행

    if process.is_alive():  # 타임아웃 초과 시 프로세스 종료
        process.terminate()
        process.join()  # 프로세스 종료를 보장
        print(f"Function timed out after {timeout} seconds.")
        return Exception("Function execution exceeded the timeout limit.")
    else:
        return queue.get()  # 결과 반환


def check_timeout_PoT(func, timeout):
    def wrapper(queue):
        result = func()
        queue.put(result)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapper, args=(queue,))
    process.start()
    process.join(timeout)  # 타임아웃 시간 동안 실행

    if process.is_alive():  # 타임아웃 초과 시 프로세스 종료
        process.terminate()
        process.join()  # 프로세스 종료를 보장
        print(f"Function timed out after {timeout} seconds.")
        return {"code": "inference timeout error"}
    else:
        return queue.get()  # 결과 반환


def check_timeout_CoT(func, timeout):
    def wrapper(queue):
        result = func()
        queue.put(result)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapper, args=(queue,))
    process.start()
    process.join(timeout)  # 타임아웃 시간 동안 실행

    if process.is_alive():  # 타임아웃 초과 시 프로세스 종료
        process.terminate()
        process.join()  # 프로세스 종료를 보장
        print(f"Function timed out after {timeout} seconds.")
        return {"solution": "inference timeout error", 'answer' : "inference timeout error"}
    else:
        return queue.get()  # 결과 반환


def check_timeout_PoT_refine(func, timeout):
    def wrapper(queue):
        result = func()
        queue.put(result)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapper, args=(queue,))
    process.start()
    process.join(timeout)  # 타임아웃 시간 동안 실행

    if process.is_alive():  # 타임아웃 초과 시 프로세스 종료
        process.terminate()
        process.join()  # 프로세스 종료를 보장
        print(f"Function timed out after {timeout} seconds.")
        return {"code": "inference timeout error"}
    else:
        return queue.get()  # 결과 반환


def check_timeout_CoT_refine(func, timeout):
    def wrapper(queue):
        result = func()
        queue.put(result)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapper, args=(queue,))
    process.start()
    process.join(timeout)  # 타임아웃 시간 동안 실행

    if process.is_alive():  # 타임아웃 초과 시 프로세스 종료
        process.terminate()
        process.join()  # 프로세스 종료를 보장
        print(f"Function timed out after {timeout} seconds.")
        return {"solution": "inference timeout error", 'answer' : "inference timeout error"}
    else:
        return queue.get()  # 결과 반환


def check_timeout_text2sql(func, timeout):
    def wrapper(queue):
        result = func()
        queue.put(result)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapper, args=(queue,))
    process.start()
    process.join(timeout)  # 타임아웃 시간 동안 실행

    if process.is_alive():  # 타임아웃 초과 시 프로세스 종료
        process.terminate()
        process.join()  # 프로세스 종료를 보장
        print(f"Function timed out after {timeout} seconds.")
        return {"code": "inference timeout error"}
    else:
        return queue.get()  # 결과 반환


def check_timeout_text2sql_refine(func, timeout):
    def wrapper(queue):
        result = func()
        queue.put(result)

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapper, args=(queue,))
    process.start()
    process.join(timeout)  # 타임아웃 시간 동안 실행

    if process.is_alive():  # 타임아웃 초과 시 프로세스 종료
        process.terminate()
        process.join()  # 프로세스 종료를 보장
        print(f"Function timed out after {timeout} seconds.")
        return {"code": "inference timeout error"}
    else:
        return queue.get()  # 결과 반환
