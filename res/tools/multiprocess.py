from multiprocessing.pool import Pool
from multiprocessing import Process
import signal


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class NoDaemonProcess(Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(Pool):
    Process = NoDaemonProcess
