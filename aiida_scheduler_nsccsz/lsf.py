"""LSF scheduler plugin for NSCCSZ HPC cluster (Platform LSF 8.0.1).

This plugin is designed for the NSCCSZ cluster which runs an older version of
Platform LSF (8.0.1, December 2011) that does not support modern bjobs flags
like -noheader or -o (custom output format), which were introduced in LSF 10+.

Key differences from the built-in core.lsf plugin:
  - Uses 'bjobs -w' instead of 'bjobs -noheader -o ...' for job listing
  - Parses classic fixed-width bjobs output columns
  - Uses 'bsub script.sh' instead of 'bsub < script.sh' (cluster esub compat)
  - Removes $LSB_OUTDIR copy logic (jobs run in submission directory)
  - Handles 'No unfinished job found' gracefully (retval 0 or 255)
"""

import re
import string

import aiida.schedulers
from aiida.common.escaping import escape_for_bash
from aiida.schedulers import SchedulerError, SchedulerParsingError
from aiida.schedulers.datastructures import JobInfo, JobState
from aiida.schedulers.plugins.lsf import LsfJobResource, LsfScheduler

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


class NsccsLsfScheduler(LsfScheduler):
    """LSF scheduler plugin for NSCCSZ HPC cluster (Platform LSF 8.0.1).

    Designed for clusters running older LSF versions that lack modern
    bjobs output formatting options.
    """

    _logger = aiida.schedulers.Scheduler._logger.getChild('nsccsz_lsf')

    # Reuse the same job resource class from the built-in LSF plugin
    _job_resource_class = LsfJobResource

    # Query only by list of jobs and not by user
    _features = {
        'can_query_by_user': False,
    }

    def _get_joblist_command(self, jobs=None, user=None):
        """Return the bjobs command to list jobs.

        Uses classic 'bjobs -w' output format compatible with LSF 8.0+.
        Does NOT use -noheader or -o flags (require LSF 10+).
        """
        from aiida.common.exceptions import FeatureNotAvailable

        command = ['bjobs', '-w']

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
        """Parse the classic fixed-width bjobs output.

        Default bjobs -w columns:
          JOBID  USER  STAT  QUEUE  FROM_HOST  EXEC_HOST  JOB_NAME  SUBMIT_TIME

        Handles:
          - 'No unfinished job found' message (retval 0 or 255)
          - Header-based column position detection for robustness
          - Multi-host EXEC_HOST values
        """
        # 'No unfinished job found' / 'No job found' is normal — not an error
        no_job_msgs = ('No unfinished job found', 'No job found')
        if retval != 0:
            if any(msg in stderr for msg in no_job_msgs):
                return []
            if any(msg in stdout for msg in no_job_msgs):
                return []
            self.logger.warning(
                f'Error in _parse_joblist_output: retval={retval}; '
                f'stdout={stdout}; stderr={stderr}'
            )
            raise SchedulerError(
                f'Error during parsing joblist output, retval={retval}\n'
                f'stdout={stdout}\nstderr={stderr}'
            )

        # Also handle retval=0 with 'No unfinished job found' in stdout
        if any(msg in stdout for msg in no_job_msgs):
            return []

        lines = stdout.splitlines()
        if not lines:
            return []

        # Find the header line (starts with JOBID)
        header_line = None
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('JOBID'):
                header_line = line
                data_start = i + 1
                break

        if header_line is None:
            return []

        # Determine column positions from header
        col_names = [
            'JOBID', 'USER', 'STAT', 'QUEUE',
            'FROM_HOST', 'EXEC_HOST', 'JOB_NAME', 'SUBMIT_TIME',
        ]
        col_positions = []
        for col_name in col_names:
            pos = header_line.find(col_name)
            if pos == -1:
                self.logger.warning(
                    f'Column {col_name} not found in bjobs header: {header_line}'
                )
                pos = None
            col_positions.append(pos)

        # Parse each job line
        job_list = []
        for line in lines[data_start:]:
            if not line.strip():
                continue

            # Extract fields using column positions
            fields = {}
            for idx, col_name in enumerate(col_names):
                start = col_positions[idx]
                if start is None:
                    fields[col_name] = ''
                    continue
                if idx + 1 < len(col_names) and col_positions[idx + 1] is not None:
                    end = col_positions[idx + 1]
                else:
                    end = len(line)
                # Guard against short lines
                if start >= len(line):
                    fields[col_name] = ''
                else:
                    fields[col_name] = line[start:end].strip()

            job_id = fields.get('JOBID', '').strip()
            if not job_id:
                continue

            this_job = JobInfo()
            this_job.job_id = job_id

            # Parse job state
            job_state_raw = fields.get('STAT', '').strip()
            try:
                this_job.job_state = _MAP_STATUS_LSF[job_state_raw]
            except KeyError:
                self.logger.warning(
                    f"Unrecognized job_state '{job_state_raw}' for job id {job_id}"
                )
                this_job.job_state = JobState.UNDETERMINED

            this_job.job_owner = fields.get('USER', '').strip()
            this_job.queue_name = fields.get('QUEUE', '').strip()

            # Execution host (only meaningful for running jobs)
            exec_host = fields.get('EXEC_HOST', '').strip()
            if exec_host and exec_host != '-' and this_job.job_state == JobState.RUNNING:
                this_job.allocated_machines_raw = exec_host

            this_job.title = fields.get('JOB_NAME', '').strip()

            # Parse submission time
            submit_time_str = fields.get('SUBMIT_TIME', '').strip()
            if submit_time_str:
                try:
                    this_job.submission_time = self._parse_time_string(
                        submit_time_str, fmt='%b %d %H:%M'
                    )
                except (ValueError, TypeError):
                    self.logger.warning(
                        f'Error parsing submission time for job id {job_id}'
                    )

            this_job.raw_data = line
            job_list.append(this_job)

        return job_list

    def _get_submit_command(self, submit_script):
        """Return the string to execute to submit a given script.

        Uses 'bsub script.sh' (script as argument) instead of 'bsub < script.sh'
        (stdin redirect) for compatibility with the NSCCSZ cluster's custom esub.
        """
        submit_command = f'bsub {submit_script}'
        self.logger.info(f'submitting with: {submit_command}')
        return submit_command

    def _get_submit_script_header(self, job_tmpl):
        """Return the submit script header with #BSUB directives.

        Simplified for NSCCSZ cluster:
          - No $LSB_OUTDIR copy logic (jobs run in submission directory)
          - No #BSUB -nnodes (not available in LSF 8.0)
        """
        lines = []

        if job_tmpl.submit_as_hold:
            lines.append('#BSUB -H')

        if job_tmpl.rerunnable:
            lines.append('#BSUB -r')
        else:
            lines.append('#BSUB -rn')

        if job_tmpl.email:
            lines.append(f'#BSUB -u {job_tmpl.email}')

        if job_tmpl.email_on_started:
            lines.append('#BSUB -B')
        if job_tmpl.email_on_terminated:
            lines.append('#BSUB -N')

        if job_tmpl.job_name:
            job_title = re.sub(r'[^a-zA-Z0-9_.-]+', '', job_tmpl.job_name)
            if not job_title or (job_title[0] not in string.ascii_letters + string.digits):
                job_title = f'j{job_title}'
            job_title = job_title[:128]
            lines.append(f'#BSUB -J "{job_title}"')

        if not job_tmpl.import_sys_environment:
            self.logger.warning('LSF scheduler cannot ignore the user environment')

        if job_tmpl.sched_output_path:
            lines.append(f'#BSUB -o {job_tmpl.sched_output_path}')

        sched_error_path = getattr(job_tmpl, 'sched_error_path', None)
        if job_tmpl.sched_join_files:
            sched_error_path = f'{job_tmpl.sched_output_path}_'
            self.logger.warning(
                'LSF scheduler does not support joining stdout and stderr; '
                f'stderr assigned to {sched_error_path}'
            )

        if sched_error_path:
            lines.append(f'#BSUB -e {job_tmpl.sched_error_path}')

        if job_tmpl.queue_name:
            lines.append(f'#BSUB -q {job_tmpl.queue_name}')

        if job_tmpl.account:
            lines.append(f'#BSUB -G {job_tmpl.account}')

        if job_tmpl.priority:
            lines.append(f'#BSUB -sp {job_tmpl.priority}')

        if not job_tmpl.job_resource:
            raise ValueError(
                'Job resources (tot_num_mpiprocs) are required for the LSF scheduler plugin'
            )

        # Always use -n (number of processors). Do NOT use -nnodes (LSF 9.1+ only).
        lines.append(f'#BSUB -n {job_tmpl.job_resource.get_tot_num_mpiprocs()}')

        if job_tmpl.job_resource.parallel_env:
            lines.append(f'#BSUB -m "{job_tmpl.job_resource.parallel_env}"')

        if job_tmpl.max_wallclock_seconds is not None:
            try:
                tot_secs = int(job_tmpl.max_wallclock_seconds)
                if tot_secs <= 0:
                    raise ValueError
            except ValueError as exc:
                raise ValueError(
                    f"max_wallclock_seconds must be a positive integer, got: "
                    f"'{job_tmpl.max_wallclock_seconds}'"
                ) from exc
            hours = tot_secs // 3600
            minutes = -(-(tot_secs % 3600) // 60)
            lines.append(f'#BSUB -W {hours:02d}:{minutes:02d}')

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
            lines.append(f'#BSUB -M {physical_memory_kb}')

        if job_tmpl.custom_scheduler_commands:
            lines.append(job_tmpl.custom_scheduler_commands)

        # No $LSB_OUTDIR copy — jobs run in the submission directory on NSCCSZ

        return '\n'.join(lines)

    def _get_submit_script_footer(self, job_tmpl):
        """Return the submit script footer.

        No $LSB_OUTDIR copy needed — jobs run in the submission directory.
        """
        return ''
