from nexus.pipeline import Job, Pipeline, Stage


def test_pipeline_import():
    assert isinstance(Pipeline, type)


def test_pipeline_single_stage_smoke():
    seen: list[str] = []

    def worker(job: Job) -> str:
        seen.append(str(job.payload))
        return f"ok:{job.payload}"

    pipe = Pipeline([Stage("smoke", worker, num_workers=1)])
    pipe.start()
    pipe.submit(Job(id="smoke-1", payload="ping"))
    pipe.join()
    pipe.stop()

    assert seen == ["ping"]
    assert len(pipe.completed) == 1
    assert pipe.completed[0].result == "ok:ping"
