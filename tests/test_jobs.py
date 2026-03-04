"""Tests for the JobManager."""

import threading
import time

import pytest

from wraval.webapp.jobs import Job, JobManager, JobStatus, JobType


class TestJobStatus:
    def test_status_values(self):
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"


class TestJobType:
    def test_type_values(self):
        assert JobType.INFERENCE == "inference"
        assert JobType.JUDGE == "judge"


class TestJob:
    def test_defaults(self):
        job = Job(
            id="abc",
            job_type=JobType.INFERENCE,
            status=JobStatus.RUNNING,
            created_at="2025-01-01T00:00:00",
            model="test-model",
            tone="witty",
        )
        assert job.error is None
        assert job.result_summary is None

    def test_all_fields(self):
        job = Job(
            id="abc",
            job_type=JobType.JUDGE,
            status=JobStatus.FAILED,
            created_at="2025-01-01T00:00:00",
            model="m",
            tone="t",
            error="boom",
            result_summary="nope",
        )
        assert job.error == "boom"
        assert job.result_summary == "nope"


class TestJobManagerStartJob:
    def test_returns_job_with_running_status(self):
        mgr = JobManager()
        done = threading.Event()

        def task(**kw):
            done.wait(timeout=2)

        job = mgr.start_job(
            JobType.INFERENCE, task, {"model": "m1", "tone": "witty"}
        )
        assert job.status == JobStatus.RUNNING
        assert job.model == "m1"
        assert job.tone == "witty"
        assert job.job_type == JobType.INFERENCE
        assert len(job.id) == 8
        done.set()

    def test_job_completes_on_success(self):
        mgr = JobManager()

        def task(**kw):
            pass

        job = mgr.start_job(JobType.INFERENCE, task, {"model": "m", "tone": "t"})
        # Wait for the background thread to finish
        time.sleep(0.2)
        assert job.status == JobStatus.COMPLETED
        assert job.result_summary == "Done"
        assert job.error is None

    def test_job_fails_on_exception(self):
        mgr = JobManager()

        def task(**kw):
            raise ValueError("something broke")

        job = mgr.start_job(JobType.JUDGE, task, {"model": "m", "tone": "t"})
        time.sleep(0.2)
        assert job.status == JobStatus.FAILED
        assert job.error == "something broke"

    def test_kwargs_passed_to_target(self):
        mgr = JobManager()
        received = {}

        def task(**kw):
            received.update(kw)

        mgr.start_job(
            JobType.INFERENCE, task, {"model": "x", "tone": "y", "extra": 42}
        )
        time.sleep(0.2)
        assert received == {"model": "x", "tone": "y", "extra": 42}

    def test_model_and_tone_default_to_empty(self):
        mgr = JobManager()

        def task(**kw):
            pass

        job = mgr.start_job(JobType.INFERENCE, task, {})
        assert job.model == ""
        assert job.tone == ""
        time.sleep(0.1)


class TestJobManagerGetJob:
    def test_returns_job_by_id(self):
        mgr = JobManager()

        def task(**kw):
            pass

        job = mgr.start_job(JobType.INFERENCE, task, {"model": "m", "tone": "t"})
        retrieved = mgr.get_job(job.id)
        assert retrieved is job

    def test_returns_none_for_unknown_id(self):
        mgr = JobManager()
        assert mgr.get_job("nonexistent") is None


class TestJobManagerConcurrency:
    def test_rejects_second_job_of_same_type(self):
        mgr = JobManager()
        blocker = threading.Event()

        def slow_task(**kw):
            blocker.wait(timeout=5)

        mgr.start_job(JobType.INFERENCE, slow_task, {"model": "m", "tone": "t"})

        with pytest.raises(RuntimeError, match="inference job is already running"):
            mgr.start_job(
                JobType.INFERENCE, slow_task, {"model": "m2", "tone": "t2"}
            )
        blocker.set()

    def test_allows_different_type_concurrently(self):
        mgr = JobManager()
        blocker = threading.Event()

        def slow_task(**kw):
            blocker.wait(timeout=5)

        mgr.start_job(JobType.INFERENCE, slow_task, {"model": "m", "tone": "t"})
        # Should not raise — different job type
        job2 = mgr.start_job(
            JobType.JUDGE, slow_task, {"model": "m2", "tone": "t2"}
        )
        assert job2.status == JobStatus.RUNNING
        blocker.set()

    def test_lock_released_after_completion(self):
        mgr = JobManager()

        def task(**kw):
            pass

        mgr.start_job(JobType.INFERENCE, task, {"model": "m", "tone": "t"})
        time.sleep(0.2)
        # Lock should be released, so a new job should work
        job2 = mgr.start_job(
            JobType.INFERENCE, task, {"model": "m2", "tone": "t2"}
        )
        assert job2.status == JobStatus.RUNNING
        time.sleep(0.1)

    def test_lock_released_after_failure(self):
        mgr = JobManager()

        def failing_task(**kw):
            raise RuntimeError("fail")

        mgr.start_job(JobType.JUDGE, failing_task, {"model": "m", "tone": "t"})
        time.sleep(0.2)
        # Lock should be released even after failure
        job2 = mgr.start_job(
            JobType.JUDGE, lambda **kw: None, {"model": "m2", "tone": "t2"}
        )
        assert job2.status == JobStatus.RUNNING
        time.sleep(0.1)


class TestJobManagerUniqueIds:
    def test_multiple_jobs_have_unique_ids(self):
        mgr = JobManager()

        def task(**kw):
            pass

        ids = set()
        for _ in range(10):
            job = mgr.start_job(
                JobType.INFERENCE, task, {"model": "m", "tone": "t"}
            )
            ids.add(job.id)
            time.sleep(0.15)  # Wait for completion so lock is released

        assert len(ids) == 10
