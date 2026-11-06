"""Cross-process lease: one input owner per game installation, no UI APIs."""
import hashlib
import os
from pathlib import Path
import tempfile


class LiveLease:
    def __init__(self,game_dir,*,directory=None):
        key=hashlib.sha256(str(Path(game_dir).resolve()).casefold().encode()).hexdigest()[:24]
        self.path=Path(directory or tempfile.gettempdir())/('bg-client-input-'+key+'.lock')
        self.stream=None

    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        stream=self.path.open('a+b')
        if self.path.stat().st_size==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:
            stream.close()
            raise RuntimeError('another_live_agent_is_running: 请先停止已有实机脚本') from None
        self.stream=stream;return self

    def __exit__(self,*exc):
        if self.stream:
            # Closing the descriptor releases the OS lease, including failures.
            self.stream.close();self.stream=None

