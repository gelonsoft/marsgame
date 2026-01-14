import torch
import queue
import threading
import time

class InferenceServer:
    def __init__(self, model, device="cuda", batch_size=256, max_wait_ms=5):
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms

        self.request_queue = queue.Queue()
        self.running = True

        self.worker = threading.Thread(target=self._loop, daemon=True)
        self.worker.start()

    def submit(self, obs, action_mask):
        response_queue = queue.Queue()
        self.request_queue.put((obs, action_mask, response_queue))
        return response_queue

    def _loop(self):
        while self.running:
            batch = []
            start = time.time()

            while len(batch) < self.batch_size:
                try:
                    item = self.request_queue.get(timeout=self.max_wait_ms / 1000)
                    batch.append(item)
                except queue.Empty:
                    break

            if not batch:
                continue

            obses, masks, reply_queues = zip(*batch)

            obs_tensor = torch.stack(obses).to(self.device)
            mask_tensor = torch.tensor(masks).to(self.device)

            with torch.no_grad():
                actions, logp, entropy, values = self.model.get_action_and_value(
                    obs_tensor, action_mask=mask_tensor
                )

            for i, q in enumerate(reply_queues):
                q.put((actions[i].cpu(), logp[i].cpu(), values[i].cpu()))
