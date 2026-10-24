"""Run a command fully detached (new session, no controlling terminal), logging to a file.

macOS has no `setsid`; jobs launched this way survive the launching shell/session exiting.
Usage:  python scripts/detach.py <logfile> <cmd> [args...]
"""
import os
import sys

log, cmd = sys.argv[1], sys.argv[2:]
if os.fork():
    sys.exit(0)
os.setsid()
if os.fork():
    sys.exit(0)
fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1)
os.dup2(fd, 2)
os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
os.execvp(cmd[0], cmd)
