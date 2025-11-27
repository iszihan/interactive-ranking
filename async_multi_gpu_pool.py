# multi_gpu_pool.py
import os
import importlib
from multiprocessing import Process, Queue, set_start_method
from queue import Empty
from itertools import count

# Good practice with CUDA + multiprocessing
try:
    set_start_method("spawn", force=True)
except RuntimeError:
    # already set
    pass


def _worker_main(gpu_id, module_name, task_queue, result_queue):
    os.environ["TQDM_DISABLE"] = "1"

    # Bind this worker to a single GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Import torch & your inference module AFTER setting CUDA_VISIBLE_DEVICES
    import torch
    # torch.cuda.set_device(0)  # within this process, "0" refers to gpu_id
    torch.cuda.set_device(int(gpu_id))

    print(
        f'Worker PID {os.getpid()} using GPU {gpu_id}, torch current = {torch.cuda.current_device()}')

    mod = importlib.import_module(module_name)

    while True:
        try:
            task = task_queue.get(timeout=1)
        except Empty:
            continue

        if task is None:
            break

        job_id, func_name, kwargs = task

        try:
            fn = getattr(mod, func_name)
            result = fn(**kwargs)
            result_queue.put((job_id, result, None))
        except Exception as e:
            # propagate error back
            result_queue.put((job_id, None, e))


class MultiGPUInferPool:
    def __init__(self, gpu_ids, module_name="parallel_infer"):
        """
        gpu_ids: list of ints, e.g. [0, 1, 2]
        module_name: Python module path where infer* functions live
        """
        self.gpu_ids = list(gpu_ids)
        self.module_name = module_name
        self.task_queue = Queue()
        self.result_queue = Queue()
        self.procs = []
        self._id_gen = count()

        # Spawn one worker per GPU
        for gid in self.gpu_ids:
            p = Process(
                target=_worker_main,
                args=(gid, self.module_name, self.task_queue, self.result_queue),
            )
            p.daemon = True
            p.start()
            self.procs.append(p)

    def submit(self, func_name, kwargs):
        job_id = next(self._id_gen)
        self.task_queue.put((job_id, func_name, kwargs))
        return job_id

    def get_result(self, timeout=None):
        return self.result_queue.get(timeout=timeout)

    def try_get_result(self):
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None

    def map(self, func_name, list_of_kwargs):
        """
        func_name: string, e.g. "infer" or "infer_img2img"
        list_of_kwargs: list[dict], each dict is kwargs for that function

        Returns:
            list of results in the same order as list_of_kwargs.
        """
        num_jobs = len(list_of_kwargs)
        if num_jobs == 0:
            return []

        job_positions = {}
        for idx, kwargs in enumerate(list_of_kwargs):
            job_id = self.submit(func_name, kwargs)
            job_positions[job_id] = idx

        results = [None] * num_jobs
        num_returned = 0
        errors = []

        while num_returned < num_jobs:
            job_id, result, err = self.result_queue.get()
            idx = job_positions[job_id]
            if err is not None:
                errors.append((job_id, err))
                results[idx] = None
            else:
                results[idx] = result
            num_returned += 1

        if errors:
            # You can make this fancier if you want
            raise RuntimeError(f"Errors in workers: {errors}")

        return results

    def shutdown(self):
        # Signal workers to exit
        for _ in self.procs:
            self.task_queue.put(None)
        for p in self.procs:
            p.join()
