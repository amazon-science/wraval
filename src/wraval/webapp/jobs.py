"""Job manager for background inference and judge tasks."""

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(str, Enum):
    INFERENCE = "inference"
    JUDGE = "judge"


@dataclass
class Job:
    id: str
    job_type: JobType
    status: JobStatus
    created_at: str
    model: str
    tone: str
    error: Optional[str] = None
    result_summary: Optional[str] = None


class JobManager:
    """In-memory job tracker with one-per-type concurrency control."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._locks = {
            "inference": threading.Lock(),
            "judge": threading.Lock(),
        }

    def start_job(self, job_type: JobType, target_fn, kwargs) -> Job:
        lock = self._locks[job_type.value]
        if not lock.acquire(blocking=False):
            raise RuntimeError(f"A {job_type.value} job is already running")

        job_id = str(uuid.uuid4())[:8]
        job = Job(
            id=job_id,
            job_type=job_type,
            status=JobStatus.RUNNING,
            created_at=datetime.utcnow().isoformat(),
            model=kwargs.get("model", ""),
            tone=kwargs.get("tone", ""),
        )
        self._jobs[job_id] = job

        def run():
            try:
                target_fn(**kwargs)
                job.status = JobStatus.COMPLETED
                job.result_summary = "Done"
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)
            finally:
                lock.release()

        threading.Thread(target=run, daemon=True).start()
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)
