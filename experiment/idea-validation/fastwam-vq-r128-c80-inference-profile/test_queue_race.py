import unittest
from queue_race import choose_winner


class RaceTests(unittest.TestCase):
    def test_both_queued_do_not_cancel(self):
        self.assertIsNone(choose_winner({'1.pbs':{'job_state':'Q'},'2.pbs':{'job_state':'Q'}}))

    def test_one_running_wins(self):
        self.assertEqual(choose_winner({'1.pbs':{'job_state':'Q'},'2.pbs':{'job_state':'R'}}),'2.pbs')

    def test_earliest_start_and_explicit_tie_break(self):
        rows={'1.pbs':{'job_state':'R','stime':20},'2.pbs':{'job_state':'R','stime':10}}
        self.assertEqual(choose_winner(rows),'2.pbs')
        rows['1.pbs']['stime']=10
        self.assertEqual(choose_winner(rows),'1.pbs')
