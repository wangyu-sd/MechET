"""Reuse real native tool-call sampling through vLLM or the working A100 HF runtime."""
from concurrent.futures import Future
from contextlib import contextmanager
from queue import Empty, Queue
from threading import Thread
import time


class BatchDispatch:
    """Serialize GPU batches while target controllers submit independent requests."""
    def __init__(self, generate, width):
        self.generate, self.width = generate, width
        self.queue = Queue()
        self.thread = Thread(target=self._work, daemon=True)
        self.thread.start()

    def submit(self, request):
        future = Future()
        self.queue.put((request, future))
        return future

    def close(self):
        self.queue.put(None)
        self.thread.join()

    def _work(self):
        error = None
        while True:
            item = self.queue.get()
            if item is None:
                break
            batch = [item]
            deadline = time.monotonic() + .02
            closing = False
            while len(batch) < self.width:
                try:
                    item = self.queue.get(timeout=max(0, deadline-time.monotonic()))
                except Empty:
                    break
                if item is None:
                    closing = True
                    break
                batch.append(item)
            try:
                if error is not None:
                    raise error
                results = self.generate([request for request, _ in batch])
                if len(results) != len(batch):
                    raise ValueError("HF sampler dropped a request")
                for (_, future), result in zip(batch, results, strict=True):
                    future.set_result(result)
            except Exception as exc:
                error = exc
                for _, future in batch:
                    future.set_exception(exc)
            if closing:
                break


@contextmanager
def loaded_actor(backend, cfg, output, version):
    from mechet.qwen35_protocol import OFFICIAL_REVISION
    from mechet.route_transformers_import import import_transformers_indexed

    if backend == "hf":
        from mechet.route_runtime_efficiency import configure_profile
        cfg["acceleration"] = dict(cfg["acceleration"], sampling_kernel="private_pinned_argmax_v1",
                                   hf_length_bucketing=True)
        configure_profile(cfg, cfg["code_root"])
    import_transformers_indexed(cfg)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["base_path"], revision=OFFICIAL_REVISION, local_files_only=True)
    journal = output / "native_generation.jsonl"
    if backend == "hf":
        import torch
        from peft import PeftModel
        from mechet.external_route_io import BatchedNativeRouteSampler, ReceiptBatch
        from mechet.qwen35_protocol import load_language_model
        from mechet.route_precision import configure_precision

        torch.cuda.set_device(0)
        configure_precision(cfg)
        model, _ = load_language_model(cfg["base_path"], dtype=torch.bfloat16)
        actor = PeftModel.from_pretrained(model, cfg["parent"], is_trainable=False).cuda().eval()
        sampler = BatchedNativeRouteSampler(actor, tokenizer, cfg, journal, ReceiptBatch(output / "executor_journal.jsonl"))
        dispatcher = BatchDispatch(sampler.generate, cfg["generation_batch_size"])
        try:
            yield dispatcher, journal, "gpu-requested"
        finally:
            dispatcher.close()
        return

    from mechet.route_vllm_backend import VllmRouteSampler, bind_rollout_adapter, start_vllm_process
    process = start_vllm_process(cfg)
    try:
        adapter = bind_rollout_adapter(cfg, version)
        import json
        with (output / "adapter_load.json").open("x") as f:
            json.dump(adapter, f)
        sampler = VllmRouteSampler(process, tokenizer, cfg, journal)
        try:
            yield sampler, journal, "vllm-submitted"
        finally:
            sampler.close()
    finally:
        process.close()
