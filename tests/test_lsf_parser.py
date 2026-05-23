import pytest

from aiida.schedulers import SchedulerError
from aiida.schedulers.datastructures import JobState

from aiida_scheduler_nsccsz.lsf import NsccsLsfScheduler


BJOBS_L_OUTPUT = """Job <9352294>, Job Name <aiida-11312>, User <nsgsx_kl>, Project <default>, Stat
                          us <PEND>, Queue <Gsx_normal>, Job Priority <50>, Com
                          mand <_aiidasubmit.sh>
Sat May 23 18:18:44 2026: Submitted from host <gsnew2010>, CWD <$HOME/work/lixi
                          ang/d5/46/376f-f288-478b-a841-bcc798091958>, Output F
                          ile <_scheduler-stdout.txt>, Error File <_scheduler-s
                          tderr.txt>, Notify when job ends, Not Re-runnable, 36
                           Processors Requested, Requested Resources < span[pti
                          le=36] >;
 PENDING REASONS:
 Job slot limit reached: 455 hosts;
------------------------------------------------------------------------------
"""


def test_parse_joblist_ignores_missing_job_stderr_with_valid_stdout():
    scheduler = NsccsLsfScheduler()

    jobs = scheduler._parse_joblist_output(
        255,
        BJOBS_L_OUTPUT,
        'Job <9331812> is not found\n',
    )

    assert len(jobs) == 1
    assert jobs[0].job_id == '9352294'
    assert jobs[0].title == 'aiida-11312'
    assert jobs[0].job_state == JobState.QUEUED
    assert jobs[0].queue_name == 'Gsx_normal'
    assert jobs[0].num_mpiprocs == 36


def test_parse_joblist_skips_missing_job_stdout_with_valid_entries():
    scheduler = NsccsLsfScheduler()

    jobs = scheduler._parse_joblist_output(
        255,
        'Job <9331812> is not found\n' + BJOBS_L_OUTPUT,
        '',
    )

    assert [job.job_id for job in jobs] == ['9352294']


def test_parse_joblist_returns_empty_for_missing_job_only():
    scheduler = NsccsLsfScheduler()

    jobs = scheduler._parse_joblist_output(
        255,
        '',
        'Job <9331812> is not found\n',
    )

    assert jobs == []


def test_parse_joblist_raises_for_real_nonzero_scheduler_error():
    scheduler = NsccsLsfScheduler()

    with pytest.raises(SchedulerError):
        scheduler._parse_joblist_output(
            255,
            BJOBS_L_OUTPUT,
            'LSF system unavailable\n',
        )
