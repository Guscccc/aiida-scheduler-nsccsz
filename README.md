# aiida-scheduler-nsccsz

AiiDA scheduler plugin for the **NSCCSZ (National Supercomputing Center in Shenzhen) HPC cluster 曙光6000集群** running **Platform LSF 8.0.1**.

## Why this plugin?

AiiDA's built-in `core.lsf` scheduler plugin uses `bjobs -noheader -o '...'` which requires LSF 10+. The NSCCSZ cluster runs Platform LSF 8.0.1 (2011) which does not support these flags. This plugin provides a compatible scheduler that works with the classic `bjobs` output format.

## Key differences from `core.lsf`

| Feature | `core.lsf` | `nsccsz.lsf` |
|---|---|---|
| Job listing | `bjobs -noheader -o '...'` | `bjobs -w` |
| Output parsing | Custom delimiter | Fixed-width columns |
| Submit command | `bsub < script.sh` | `bsub script.sh` |
| Script header | Includes `$LSB_OUTDIR` copy | No copy (jobs run in place) |
| Node count | `#BSUB -nnodes` (LSF 9.1+) | Always uses `#BSUB -n` |

## Wall-clock limits

AiiDA `metadata.options.max_wallclock_seconds` is converted to the cluster's
native `bsub -W [hour:]minute` command-line option. Positive partial minutes
are rounded up because LSF has minute resolution. NSCCSZ currently has
`ABS_RUNLIMIT=Y`, so this is an absolute wall-clock limit. At the limit LSF
sends `SIGUSR2` and may kill a job that has not exited within the configured
grace period.

The scheduler option does not add application-specific command-line flags;
calculation plugins remain responsible for any separate graceful-stop option.

## Installation

```bash
pip install aiida-scheduler-nsccsz
```

Or install in development mode:

```bash
git clone https://github.com/guscccc/aiida-scheduler-nsccsz.git
cd aiida-scheduler-nsccsz
pip install -e .
```

## Usage

### 1. Set up the computer

```bash
verdi computer setup \
    --label [label]\
    --hostname [hostname]\
    --transport core.ssh_async\
    --scheduler nsccsz.lsf \
    --work-dir '/path/to/workdir'
```

### 2. Configure SSH

```bash
verdi computer configure core.ssh_async nsccsz
```

### 3. Test

```bash
verdi computer test nsccsz
```

## SSH Configuration Example

```sshconfig
# Login node (external access via SOCKS proxy)
Host nsccsz_login
    HostName ...
    User your_username
    HostKeyAlgorithms +ssh-rsa
    PubkeyAcceptedAlgorithms +ssh-rsa
    ProxyCommand nc -X 5 -x ip_address:port %h %p

# Compile/submit node (internal, reached via login node)
Host nsccsz_compile
    HostName ...
    User your_username
    HostKeyAlgorithms +ssh-rsa
    PubkeyAcceptedAlgorithms +ssh-rsa
    ProxyJump nsccsz_login
```

## License

MIT
