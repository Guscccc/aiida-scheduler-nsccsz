from types import SimpleNamespace

import pytest

from aiida.schedulers import SchedulerError
from aiida.schedulers.datastructures import JobState

from aiida_scheduler_nsccsz.lsf import NsccsLsfJobResource, NsccsLsfScheduler


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


def _job_template(resources, *, max_wallclock_seconds=None, custom_scheduler_commands=None):
    return SimpleNamespace(
        account=None,
        custom_scheduler_commands=custom_scheduler_commands,
        email=None,
        email_on_started=False,
        email_on_terminated=True,
        import_sys_environment=True,
        job_name='aiida-11908',
        job_resource=NsccsLsfJobResource(**resources),
        max_memory_kb=None,
        max_wallclock_seconds=max_wallclock_seconds,
        priority=None,
        queue_name='Gsx_normal',
        rerunnable=False,
        sched_error_path='_scheduler-stderr.txt',
        sched_join_files=False,
        sched_output_path='_scheduler-stdout.txt',
        submit_as_hold=False,
    )


def test_submit_header_preserves_ranks_per_node_as_bsub_span_ptile():
    scheduler = NsccsLsfScheduler()
    header = scheduler._get_submit_script_header(
        _job_template({'tot_num_mpiprocs': 36, 'num_mpiprocs_per_machine': 18})
    )

    assert '# The #AIIDA_LSF_ARG lines below are plugin metadata' in header
    assert '#AIIDA_LSF_ARG -n 36' in header
    assert '#AIIDA_LSF_ARG -R "span[ptile=18]"' in header
    assert 'NP=36' in header
    assert 'NP_PER_NODE=18' in header

    command = scheduler._build_submit_command_from_script('_aiidasubmit.sh', header)
    assert command == (
        'bsub -rn -N -J "aiida-11908" -o _scheduler-stdout.txt '
        '-e _scheduler-stderr.txt -q Gsx_normal -n 36 '
        '-R "span[ptile=18]" _aiidasubmit.sh'
    )


def test_submit_header_defaults_to_full_node_ptile_for_legacy_tot_only_resources():
    scheduler = NsccsLsfScheduler()
    header = scheduler._get_submit_script_header(_job_template({'tot_num_mpiprocs': 36}))

    assert '#AIIDA_LSF_ARG -n 36' in header
    assert '#AIIDA_LSF_ARG -R "span[ptile=36]"' in header
    assert 'NP_PER_NODE=36' in header


@pytest.mark.parametrize(
    'seconds,expected',
    [
        (1, '00:01'),
        (59, '00:01'),
        (60, '00:01'),
        (61, '00:02'),
        (3599, '01:00'),
        (3600, '01:00'),
        (3601, '01:01'),
        (43200, '12:00'),
        (86399, '24:00'),
        (86400, '24:00'),
    ],
)
def test_submit_header_converts_wallclock_to_bsub_runlimit(seconds, expected):
    """Wallclock seconds become one canonical ``bsub -W`` option."""
    scheduler = NsccsLsfScheduler()
    header = scheduler._get_submit_script_header(
        _job_template(
            {'tot_num_mpiprocs': 36},
            max_wallclock_seconds=seconds,
        )
    )

    assert header.count(f'#AIIDA_LSF_ARG -W {expected}') == 1
    command = scheduler._build_submit_command_from_script('_aiidasubmit.sh', header)
    assert command.count(f'-W {expected}') == 1


@pytest.mark.parametrize('seconds', [0, -1])
def test_submit_header_rejects_invalid_explicit_wallclock(seconds):
    """Explicit non-positive wallclock values are invalid."""
    scheduler = NsccsLsfScheduler()
    with pytest.raises(ValueError, match='positive integer'):
        scheduler._get_submit_script_header(
            _job_template(
                {'tot_num_mpiprocs': 36},
                max_wallclock_seconds=seconds,
            )
        )


def test_parse_joblist_extracts_requested_wallclock():
    """Parse the RUNLIMIT representation returned by ``bjobs -l``."""
    scheduler = NsccsLsfScheduler()
    output = BJOBS_L_OUTPUT.replace(
        ' PENDING REASONS:',
        ' RUNLIMIT 720.0 min;\n PENDING REASONS:',
    )

    jobs = scheduler._parse_joblist_output(0, output, '')

    assert len(jobs) == 1
    assert jobs[0].requested_wallclock_time_seconds == 43200


def test_job_resource_accepts_legacy_lsf_resource_keys():
    resource = NsccsLsfJobResource(
        tot_num_mpiprocs=36,
        num_machines=2,
        use_num_machines=True,
        parallel_env='hostA hostB',
    )

    assert resource.num_machines == 2
    assert resource.num_mpiprocs_per_machine == 18
    assert resource.parallel_env == 'hostA hostB'


def test_job_resource_uses_default_mpiprocs_per_machine_when_needed():
    resource = NsccsLsfJobResource(
        tot_num_mpiprocs=36,
        default_mpiprocs_per_machine=12,
    )

    assert resource.num_machines == 3
    assert resource.num_mpiprocs_per_machine == 12
