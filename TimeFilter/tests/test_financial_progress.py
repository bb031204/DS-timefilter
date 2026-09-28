"""Progress events stay live in the terminal and concise in the log."""
import io
import unittest

from utils.financial_progress import FinancialProgress, PROGRESS_PREFIX, format_duration


class TtyBuffer(io.StringIO):
    def isatty(self):
        return True


class FinancialProgressTests(unittest.TestCase):
    def test_interactive_epoch_bar_and_times(self):
        now = [100]
        stream, log = TtyBuffer(), io.StringIO()
        progress = FinancialProgress(log, stream=stream, clock=lambda: now[0])
        progress.update(f'{PROGRESS_PREFIX}1\t2\t0\t4\n')
        now[0] = 10 + 100
        progress.update(f'{PROGRESS_PREFIX}1\t2\t2\t4\n')
        self.assertIn('used 00:10 | left 00:10', stream.getvalue())
        self.assertIn('ETA 00:30', stream.getvalue())
        now[0] = 120
        progress.update(f'{PROGRESS_PREFIX}1\t2\t4\t4\n')
        self.assertEqual(log.getvalue().count('Epoch'), 1)
        self.assertIn('Epoch   1/2', log.getvalue())
        self.assertNotIn(PROGRESS_PREFIX, log.getvalue())
        self.assertIn('\r', stream.getvalue())

    def test_noninteractive_output_has_one_line_per_epoch(self):
        stream, log = io.StringIO(), io.StringIO()
        progress = FinancialProgress(log, stream=stream, clock=lambda: 10)
        for step in range(4):
            progress.update(f'{PROGRESS_PREFIX}1\t1\t{step}\t3\n')
        self.assertEqual(stream.getvalue().count('\n'), 1)
        self.assertEqual(log.getvalue().count('\n'), 1)
        self.assertEqual(format_duration(3661), '1:01:01')


if __name__ == '__main__':
    unittest.main()
