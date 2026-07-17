"""LSF scheduler plugin for NSCCSZ HPC cluster (Platform LSF 8.0.1).

This plugin is designed for the NSCCSZ cluster which runs an older version of
Platform LSF (8.0.1, December 2011) that does not support modern bjobs flags
like -noheader or -o (custom output format), which were introduced in LSF 10+.

Key differences from the built-in core.lsf plugin:
  - Uses 'bjobs -l' instead of 'bjobs -noheader -o ...' for job listing
  - Does NOT use `#BSUB` directives in submit scripts but use command line arguments with `bsub` command
  - Parses multi-line bjobs -l output format with full timestamps
  - Uses 'bsub script.sh' instead of 'bsub < script.sh' (cluster esub compat)
  - Removes $LSB_OUTDIR copy logic (jobs run in submission directory)
  - Handles 'No unfinished job found' gracefully (retval 0 or 255)
"""

import re
import string
from datetime import datetime

import aiida.schedulers
from aiida.common.escaping import escape_for_bash
from aiida.schedulers import SchedulerError, SchedulerParsingError
from aiida.schedulers.datastructures import JobInfo, JobState, NodeNumberJobResource
from aiida.schedulers.plugins.lsf import LsfScheduler

# LSF status code mapping (same across LSF versions)
_MAP_STATUS_LSF = {
    'PEND': JobState.QUEUED,
    'PROV': JobState.QUEUED,
    'PSUSP': JobState.QUEUED_HELD,
    'USUSP': JobState.SUSPENDED,
    'SSUSP': JobState.SUSPENDED,
    'RUN': JobState.RUNNING,
    'DONE': JobState.DONE,
    'EXIT': JobState.DONE,
    'UNKWN': JobState.UNDETERMINED,
    'WAIT': JobState.QUEUED,
    'ZOMBI': JobState.UNDETERMINED,
}

_NO_JOB_PATTERNS = (
    re.compile(r'No unfinished job found'),
    re.compile(r'No job found'),
    re.compile(r'Job <[^>]+> is not found'),
)


def _seconds_to_lsf_runlimit(seconds):
    """Convert seconds to canonical LSF ``-W [hour:]minute`` syntax.

    LSF accepts minute resolution only, so a positive partial minute is rounded
    up to avoid requesting less wall time than AiiDA specified.
    """
    try:
        total_seconds = int(seconds)
        if total_seconds <= 0:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"max_wallclock_seconds must be a positive integer, got: '{seconds}'"
        ) from exc

    total_minutes = -(-total_seconds // 60)
    hours, minutes = divmod(total_minutes, 60)
    return f'{hours:02d}:{minutes:02d}'


class NsccsLsfJobResource(NodeNumberJobResource):
    """LSF resource class that preserves NSCCSZ ranks-per-node placement."""

    @classmethod
    def validate_resources(cls, **kwargs):
        parallel_env = kwargs.pop('parallel_env', '')
        if not isinstance(parallel_env, str):
            raise TypeError('`parallel_env` must be a string')

        kwargs.pop('use_num_machines', False)
        default_mpiprocs_per_machine = kwargs.pop('default_mpiprocs_per_machine', None)
        if 'num_mpiprocs_per_machine' not in kwargs and 'num_machines' not in kwargs:
            kwargs['num_mpiprocs_per_machine'] = (
                36 if default_mpiprocs_per_machine is None else default_mpiprocs_per_machine
            )

        resources = super().validate_resources(**kwargs)
        resources.parallel_env = parallel_env
        return resources


class NsccsLsfScheduler(LsfScheduler):
    """LSF scheduler plugin for NSCCSZ HPC cluster (Platform LSF 8.0.1).

    Designed for clusters running older LSF versions that lack modern
    bjobs output formatting options and where #BSUB directives may not
    be properly parsed by custom esub wrappers.
    """

    _logger = aiida.schedulers.Scheduler._logger.getChild('nsccsz_lsf')

    _job_resource_class = NsccsLsfJobResource

    # Query only by list of jobs and not by user
    _features = {
        'can_query_by_user': False,
    }

    @staticmethod
    def _is_missing_job_message(line):
        """Return whether a line is a benign missing-job diagnostic."""
        return any(pattern.search(line) for pattern in _NO_JOB_PATTERNS)

    @classmethod
    def _has_job_entries(cls, output):
        """Return whether ``bjobs -l`` output contains at least one job entry."""
        return any(
            line.strip().startswith('Job <') and not cls._is_missing_job_message(line)
            for line in output.splitlines()
        )

    @classmethod
    def _has_only_missing_job_messages(cls, output):
        """Return whether output consists only of benign missing-job diagnostics."""
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        return bool(lines) and all(cls._is_missing_job_message(line) for line in lines)

    def _get_joblist_command(self, jobs=None, user=None):
        """Return the bjobs command to list jobs.

        Uses 'bjobs -l' for detailed multi-line output with full timestamps.
        Does NOT use -noheader or -o flags (require LSF 10+).
        """
        from aiida.common.exceptions import FeatureNotAvailable

        command = ['bjobs', '-l']

        if user and jobs:
            raise FeatureNotAvailable('Cannot query by user and job(s) in LSF')

        if user:
            command.append(f'-u{user}')

        if jobs:
            if isinstance(jobs, str):
                command.append(jobs)
            elif isinstance(jobs, (tuple, list)):
                command.append(' '.join(jobs))
            else:
                raise TypeError(
                    "If provided, the 'jobs' variable must be a string or a list of strings"
                )

        comm = ' '.join(command)
        self.logger.debug(f'bjobs command: {comm}')
        return comm

    def _parse_joblist_output(self, retval, stdout, stderr):
        """Parse the multi-line bjobs -l output.

        Format example:
          Job <8977273>, User <nsgsx_kl>, Project <default>, Status <RUN>, Queue <Gsx_normal>, ...
          Wed Mar 11 18:17:42 2026: Submitted from host <gsnew2010>, CWD <...>, ...
          Wed Mar 11 19:35:32 2026: Started on 36 Hosts/Processors <36*gsnew2135>, ...

        Handles:
          - 'No unfinished job found' message (retval 0 or 255)
          - Multi-line job entries separated by blank lines or dashes
          - Full timestamps with year
        """
        # Missing jobs are normal when polling a batch of jobs: LSF may return
        # retval=255 even while stdout still contains valid entries for the
        # other requested job ids.
        if retval != 0:
            stdout_has_jobs = self._has_job_entries(stdout)
            stderr_is_only_missing_jobs = self._has_only_missing_job_messages(stderr)
            stdout_is_only_missing_jobs = self._has_only_missing_job_messages(stdout)

            if stdout_is_only_missing_jobs and not stderr.strip():
                return []

            if stdout_has_jobs and (not stderr.strip() or stderr_is_only_missing_jobs):
                self.logger.debug(
                    'Ignoring missing-job diagnostics from bjobs while parsing '
                    f'available job entries: stdout={stdout.strip()}; stderr={stderr.strip()}'
                )
            elif stderr_is_only_missing_jobs:
                return []
            else:
                self.logger.warning(
                    f'Error in _parse_joblist_output: retval={retval}; '
                    f'stdout={stdout}; stderr={stderr}'
                )
                raise SchedulerError(
                    f'Error during parsing joblist output, retval={retval}\n'
                    f'stdout={stdout}\nstderr={stderr}'
                )

        # Also handle retval=0 with 'No unfinished job found' in stdout
        if self._has_only_missing_job_messages(stdout):
            return []

        lines = stdout.splitlines()
        if not lines:
            return []

        # Parse multi-line job entries
        job_list = []
        current_job_lines = []
        
        for line in lines:
            if self._is_missing_job_message(line):
                continue

            # Job entries start with "Job <ID>"
            if line.strip().startswith('Job <'):
                # Process previous job if exists
                if current_job_lines:
                    job_info = self._parse_single_job_entry(current_job_lines)
                    if job_info:
                        job_list.append(job_info)
                # Start new job
                current_job_lines = [line]
            elif line.strip().startswith('---'):
                # Separator between jobs
                if current_job_lines:
                    job_info = self._parse_single_job_entry(current_job_lines)
                    if job_info:
                        job_list.append(job_info)
                    current_job_lines = []
            elif current_job_lines:
                # Continuation of current job
                current_job_lines.append(line)
        
        # Process last job
        if current_job_lines:
            job_info = self._parse_single_job_entry(current_job_lines)
            if job_info:
                job_list.append(job_info)

        return job_list

    def _parse_single_job_entry(self, lines):
        """Parse a single job entry from bjobs -l output.
        
        Args:
            lines: List of lines for a single job entry
            
        Returns:
            JobInfo object or None if parsing fails
        """
        if not lines:
            return None
            
        # Reconstruct the original text by joining with newlines and removing
        # the LSF line-wrap artifact (a newline followed by many spaces).
        raw_text = '\n'.join(lines)
        text = re.sub(r'\n\s{10,}', '', raw_text)
        
        # Extract job ID from first line: "Job <8977273>, ..."
        job_id_match = re.search(r'Job <(\d+)>', text)
        if not job_id_match:
            return None
        
        job_id = job_id_match.group(1)
        this_job = JobInfo()
        this_job.job_id = job_id
        
        # Extract user: "User <nsgsx_kl>"
        user_match = re.search(r'User <([^>]+)>', text)
        if user_match:
            this_job.job_owner = user_match.group(1)
        
        # Extract status: "Status <RUN>"
        status_match = re.search(r'Status <([^>]+)>', text)
        if status_match:
            job_state_raw = status_match.group(1)
            try:
                this_job.job_state = _MAP_STATUS_LSF[job_state_raw]
            except KeyError:
                self.logger.warning(
                    f"Unrecognized job_state '{job_state_raw}' for job id {job_id}"
                )
                this_job.job_state = JobState.UNDETERMINED
        
        # Extract queue: "Queue <Gsx_normal>"
        queue_match = re.search(r'Queue <([^>]+)>', text)
        if queue_match:
            this_job.queue_name = queue_match.group(1)
        
        # Extract job name: "Job Name <aiida-282>"
        job_name_match = re.search(r'Job Name <([^>]+)>', text)
        if job_name_match:
            this_job.title = job_name_match.group(1)
        
        # Extract number of processors: "36 Processors Requested". LSF may
        # wrap this as "36\n    Processors", which is unwrapped above.
        procs_match = re.search(r'(\d+)\s*Processors?\s+Requested', text)
        if procs_match:
            this_job.num_mpiprocs = int(procs_match.group(1))
        
        # Extract runtime limit: "30.0 min" or "RUNLIMIT 30.0 min"
        runlimit_match = re.search(r'RUNLIMIT\s+(\d+(?:\.\d+)?)\s+(min|hour)', text)
        if runlimit_match:
            time_value = float(runlimit_match.group(1))
            time_unit = runlimit_match.group(2)
            if time_unit == 'min':
                this_job.requested_wallclock_time_seconds = int(time_value * 60)
            elif time_unit == 'hour':
                this_job.requested_wallclock_time_seconds = int(time_value * 3600)
        
        # Extract submission time: "Wed Mar 11 18:17:42 2026: Submitted from host"
        # Format: Day Mon DD HH:MM:SS YYYY
        submit_match = re.search(r'(\w{3}\s+\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4}):\s+Submitted from host', text)
        if submit_match:
            submit_time_str = submit_match.group(1)
            try:
                # Parse directly with datetime since we have the full timestamp with year
                this_job.submission_time = datetime.strptime(submit_time_str, '%a %b %d %H:%M:%S %Y')
            except (ValueError, TypeError) as e:
                self.logger.warning(
                    f'Error parsing submission time "{submit_time_str}" for job id {job_id}: {e}'
                )
        
        # Extract execution host for running jobs: "Started on 36 Hosts/Processors <36*gsnew2135>"
        if this_job.job_state == JobState.RUNNING:
            exec_match = re.search(r'Started on.*?<([^>]+)>', text)
            if exec_match:
                this_job.allocated_machines_raw = exec_match.group(1)
        
        this_job.raw_data = '\n'.join(lines)
        return this_job

    def _get_submit_command(self, submit_script):
        """Return the fallback submit command.

        The actual NSCCSZ submission path is implemented in ``submit_job``,
        which reads internal AiiDA metadata lines from the generated script and
        converts them to command-line arguments. No ``#BSUB`` lines are kept in
        the final script to avoid confusion on this cluster.
        """
        submit_command = f'bsub {submit_script}'
        self.logger.info(f'submitting with fallback command: {submit_command}')
        return submit_command

    def _build_submit_command_from_script(self, submit_script, script_content):
        """Build ``bsub`` command from internal metadata lines in the script."""
        options = []

        for line in script_content.splitlines():
            stripped = line.strip()
            if stripped.startswith('#AIIDA_LSF_ARG '):
                options.append(stripped[len('#AIIDA_LSF_ARG '):].strip())

        submit_command = f'bsub {" ".join(options)} {submit_script}' if options else f'bsub {submit_script}'
        self.logger.info(f'submitting with: {submit_command}')
        return submit_command

    def submit_job(self, working_directory: str, filename: str):
        """Submit a job using command-line ``bsub`` options extracted from the script."""
        escaped_filename = escape_for_bash(filename)
        retval, stdout, stderr = self.transport.exec_command_wait(
            f'cat {escaped_filename}', workdir=working_directory
        )

        if retval != 0:
            raise SchedulerError(
                f'Unable to read submit script `{filename}` before submission. '
                f'retval={retval}; stdout={stdout}; stderr={stderr}'
            )

        submit_command = self._build_submit_command_from_script(escaped_filename, stdout)
        result = self.transport.exec_command_wait(submit_command, workdir=working_directory)
        return self._parse_submit_output(*result)

    def _get_submit_script_header(self, job_tmpl):
        """Return the submit script header without any ``#BSUB`` directives.

        Submission options are encoded as internal metadata comments and then
        converted into command-line arguments in :meth:`submit_job`.
        """
        lines = [
            '# The #AIIDA_LSF_ARG lines below are plugin metadata, not LSF directives.',
            '# Add them as bsub command-line options; simply bsub-ing this script will not apply them.',
        ]

        if job_tmpl.submit_as_hold:
            lines.append('#AIIDA_LSF_ARG -H')

        if job_tmpl.rerunnable:
            lines.append('#AIIDA_LSF_ARG -r')
        else:
            lines.append('#AIIDA_LSF_ARG -rn')

        if job_tmpl.email:
            lines.append(f'#AIIDA_LSF_ARG -u {job_tmpl.email}')

        if job_tmpl.email_on_started:
            lines.append('#AIIDA_LSF_ARG -B')
        if job_tmpl.email_on_terminated:
            lines.append('#AIIDA_LSF_ARG -N')

        if job_tmpl.job_name:
            job_title = re.sub(r'[^a-zA-Z0-9_.-]+', '', job_tmpl.job_name)
            if not job_title or (job_title[0] not in string.ascii_letters + string.digits):
                job_title = f'j{job_title}'
            job_title = job_title[:128]
            lines.append(f'#AIIDA_LSF_ARG -J "{job_title}"')

        if not job_tmpl.import_sys_environment:
            self.logger.warning('LSF scheduler cannot ignore the user environment')

        if job_tmpl.sched_output_path:
            lines.append(f'#AIIDA_LSF_ARG -o {job_tmpl.sched_output_path}')

        sched_error_path = getattr(job_tmpl, 'sched_error_path', None)
        if job_tmpl.sched_join_files:
            sched_error_path = f'{job_tmpl.sched_output_path}_'
            self.logger.warning(
                'LSF scheduler does not support joining stdout and stderr; '
                f'stderr assigned to {sched_error_path}'
            )

        if sched_error_path:
            lines.append(f'#AIIDA_LSF_ARG -e {sched_error_path}')

        if job_tmpl.queue_name:
            lines.append(f'#AIIDA_LSF_ARG -q {job_tmpl.queue_name}')

        if job_tmpl.account:
            lines.append(f'#AIIDA_LSF_ARG -G {job_tmpl.account}')

        if job_tmpl.priority:
            lines.append(f'#AIIDA_LSF_ARG -sp {job_tmpl.priority}')

        if not job_tmpl.job_resource:
            raise ValueError(
                'Job resources (tot_num_mpiprocs) are required for the LSF scheduler plugin'
            )

        tot_procs = job_tmpl.job_resource.get_tot_num_mpiprocs()
        np_per_node = int(job_tmpl.job_resource.num_mpiprocs_per_machine)

        lines.append(f'#AIIDA_LSF_ARG -n {tot_procs}')
        lines.append(f'#AIIDA_LSF_ARG -R "span[ptile={np_per_node}]"')

        parallel_env = getattr(job_tmpl.job_resource, 'parallel_env', '')
        if parallel_env:
            lines.append(f'#AIIDA_LSF_ARG -m "{parallel_env}"')

        if job_tmpl.max_wallclock_seconds is not None:
            runlimit = _seconds_to_lsf_runlimit(job_tmpl.max_wallclock_seconds)
            lines.append(f'#AIIDA_LSF_ARG -W {runlimit}')

        if job_tmpl.max_memory_kb:
            try:
                physical_memory_kb = int(job_tmpl.max_memory_kb)
                if physical_memory_kb <= 0:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(
                    f'max_memory_kb must be a positive integer (in kB), '
                    f'got: `{job_tmpl.max_memory_kb}`'
                ) from exc
            lines.append(f'#AIIDA_LSF_ARG -M {physical_memory_kb}')

        if job_tmpl.custom_scheduler_commands:
            lines.append(job_tmpl.custom_scheduler_commands)

        lines.append('')
        app_name = job_tmpl.queue_name if job_tmpl.queue_name else 'Gsx_normal'
        lines.append(f'APP_NAME={app_name}')

        lines.append(f'NP={tot_procs}')
        lines.append(f'NP_PER_NODE={np_per_node}')

        lines.append('RUN="RAW"')
        lines.append('')

        return '\n'.join(lines)

    def _get_submit_script_footer(self, job_tmpl):
        """Return the submit script footer.

        No $LSB_OUTDIR copy needed — jobs run in the submission directory.
        """
        return ''
