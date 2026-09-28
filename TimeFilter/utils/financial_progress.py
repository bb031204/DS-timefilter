"""Compact per-epoch progress for the financial subprocess launcher."""
import sys
import time


PROGRESS_PREFIX = '__TIMEFILTER_PROGRESS__\t'


def emit_progress(epoch, epochs, step, steps):
    """Send a newline-terminated event so the launcher can update immediately."""
    print(f'{PROGRESS_PREFIX}{epoch}\t{epochs}\t{step}\t{steps}', flush=True)


def format_duration(seconds):
    if seconds is None:
        return '--:--'
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f'{hours:d}:{minutes:02d}:{seconds:02d}' if hours else f'{minutes:02d}:{seconds:02d}'


class FinancialProgress:
    def __init__(self, log, stream=None, clock=None):
        self.log = log
        self.stream = stream or sys.stdout
        self.clock = clock or time.monotonic
        self.interactive = self.stream.isatty()
        self.first = None
        self.epoch_start = None
        self.epoch = None
        self.visible = False
        self.last_width = 0

    def clear_line(self):
        if self.visible:
            self.stream.write('\n')
            self.stream.flush()
            self.visible = False
            self.last_width = 0

    def update(self, line):
        epoch, epochs, step, steps = (int(value) for value in line[len(PROGRESS_PREFIX):].split())
        if not (1 <= epoch <= epochs and 0 <= step <= steps and steps > 0):
            raise ValueError('Invalid financial progress event')
        now = self.clock()
        if self.first is None:
            self.first = now
        if self.epoch != epoch:
            self.clear_line()
            self.epoch = epoch
            self.epoch_start = now
        epoch_elapsed = now - self.epoch_start
        total_elapsed = now - self.first
        epoch_left = epoch_elapsed / step * (steps - step) if step else None
        done = (epoch - 1) * steps + step
        total_left = total_elapsed / done * (epochs * steps - done) if done else None
        filled = 20 * step // steps
        bar = '#' * filled + '-' * (20 - filled)
        message = (f'Epoch {epoch:>3}/{epochs:<3} [{bar}] {step:>3}/{steps:<3} '
                   f'| used {format_duration(epoch_elapsed)} | left {format_duration(epoch_left)} '
                   f'| total {format_duration(total_elapsed)} | ETA {format_duration(total_left)}')
        if self.interactive:
            self.stream.write('\r' + message + ' ' * max(0, self.last_width - len(message)))
            self.stream.flush()
            self.visible = True
            self.last_width = len(message)
        if step == steps:
            if self.interactive:
                self.clear_line()
            else:
                self.stream.write(message + '\n')
                self.stream.flush()
            self.log.write(message + '\n')
