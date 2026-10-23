"""Small contract checks for the new arm and meaningful gripper event timing."""
import copy
import unittest
import worker


def sample(signs):
    return {'commands': [[0]*6+[v] for v in signs+[signs[-1]]*22]}


class Contracts(unittest.TestCase):
    def test_release_delay_and_missing_release(self):
        teacher = sample([1]*8+[-1]*2)
        delayed = worker.transitions(sample([1]*9+[-1]), teacher)
        self.assertTrue(delayed['direction_sequence_equal'])
        self.assertEqual(delayed['ordered_event_delays_when_sequence_equal'], [1])
        missing = worker.transitions(sample([1]*10), teacher)
        self.assertEqual(missing['missed_open_events'], 1)
        self.assertFalse(missing['direction_sequence_equal'])
        self.assertEqual(missing['ordered_event_delays_when_sequence_equal'], [])

    def test_opposite_direction_is_not_a_delay(self):
        result = worker.transitions(sample([-1]*8+[1]*2), sample([1]*8+[-1]*2))
        self.assertFalse(result['direction_sequence_equal'])
        self.assertEqual(result['extra_close_events'], 1)
        self.assertEqual(result['missed_open_events'], 1)

    def test_seed_and_checkpoint_binding(self):
        row = {'case_id':2001,'suite':'libero_spatial','sampler_seeds':[910004002,910004003]}
        p = {'protocol':'test','channel':{'deployment':'final80.pt'}}
        record = {**worker.binding(p,row,'base'),'pbs_jobid':'test','training_updates':0,
                  'action':[[0.0]*7 for _ in range(32)],'commands':[[0.0]*7 for _ in range(32)]}
        worker.validate_new(record,p,row,'base')
        bad = copy.deepcopy(record);bad['sampler_seed'] += 1
        with self.assertRaises(ValueError):worker.validate_new(bad,p,row,'base')
        bad = copy.deepcopy(record);bad['source_checkpoint'] = 'wrong.pt'
        with self.assertRaises(ValueError):worker.validate_new(bad,p,row,'base')


if __name__=='__main__':unittest.main()
