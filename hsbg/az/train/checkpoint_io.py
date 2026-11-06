"""发布已提交快照的别名；不重复序列化，不就地修改任何已发布文件。"""
import os
from pathlib import Path
import shutil
import tempfile


def publish_checkpoint(source, target):
    """同卷优先硬链接；不支持时复制。临时文件替换保持旧别名可读。"""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=target.name+'.publish-', dir=target.parent)
    os.close(fd)
    os.unlink(temp)
    try:
        try:
            os.link(source, temp)
            mode = 'hardlink'
        except OSError:
            shutil.copyfile(source, temp)
            mode = 'copy'
        os.replace(temp, target)
        return mode
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
