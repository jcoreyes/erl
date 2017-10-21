import unittest

from railrl.data_management.her_replay_buffer import HerReplayBuffer


class TestSimpleReplayBuffer(unittest.TestCase):
    """
    Some white-box tests. Black-box testing just gets too ugly since the
    interface is random (literally just random_batch()).

    For these tests let "A" mean a sample from the first trajectory, "B" mean a
    sample from the second trajectory, "C" mean ...
    """
    def assertSameContent(self, lst1, lst2):
        self.assertSetEqual(set(lst1), set(lst2))

    def test_wrapped_indices_are_not_valid(self):
        """
        Index:   0 1 2 3 4 5
        Content: B A A A B B
        Top:       X

        So the only valid transitions should be 1 and 2.
        """
        buffer = HerReplayBuffer(6, 1, 1)
        buffer.add_sample(1, 1, 1, False)  # A
        buffer.add_sample(1, 1, 1, False)  # A
        buffer.add_sample(1, 1, 1, True)   # A
        buffer.terminate_episode(1, True)  # A
        buffer.add_sample(1, 1, 1, False)  # B
        buffer.add_sample(1, 1, 1, True)   # B
        buffer.terminate_episode(1, True)  # B
        self.assertEqual(buffer._top, 1)
        self.assertSameContent(buffer._valid_transition_indices, [1, 2])

        buffer = HerReplayBuffer(6, 1, 1)
        buffer.add_sample(1, 1, 1, False)   # A
        buffer.add_sample(1, 1, 1, False)   # A
        buffer.add_sample(1, 1, 1, False)   # A
        buffer.terminate_episode(1, False)  # A
        buffer.add_sample(1, 1, 1, False)   # B
        buffer.add_sample(1, 1, 1, False)   # B
        buffer.add_sample(1, 1, 1, False)   # B
        self.assertEqual(buffer._top, 1)
        self.assertSameContent(buffer._valid_transition_indices, [1, 2])

    def test_wrapped_indices_twice(self):
        """
        Start with

        Index:   0 1 2 3 4 5
        Content: C A A B B C
        Top:       ^
        """
        buffer = HerReplayBuffer(6, 1, 1)
        buffer.add_sample(1, 1, 1, False)  # A
        buffer.add_sample(1, 1, 1, True)   # A
        buffer.terminate_episode(1, True)  # A
        buffer.add_sample(1, 1, 1, True)   # B
        buffer.terminate_episode(1, True)  # B
        buffer.add_sample(1, 1, 1, False)  # C
        buffer.terminate_episode(1, True)  # C

        self.assertEqual(buffer._top, 1)
        self.assertSameContent(buffer._valid_transition_indices, [1, 3])

        """
        but then add some samples to get:

        Index:   0 1 2 3 4 5
        Content: C D D E E F
        Top:     ^

        So the only valid transitions should be 1 and 3.
        """
        buffer.add_sample(1, 1, 1, False)   # D
        buffer.terminate_episode(1, False)  # D
        buffer.add_sample(1, 1, 1, False)   # E
        buffer.terminate_episode(1, False)  # E
        buffer.add_sample(1, 1, 1, False)   # F
        self.assertEqual(buffer._top, 0)
        self.assertSameContent(buffer._valid_transition_indices, [1, 3])

    def test_do_no_make_transition_valid_until_end_of_episode(self):
        """
        Index:   0 1 2 3 4 5
        Content: A A A A
        Top:             X

        So no transitions are valid
        """
        buffer = HerReplayBuffer(6, 1, 1)
        buffer.add_sample(1, 1, 1, False)
        buffer.add_sample(1, 1, 1, False)
        buffer.add_sample(1, 1, 1, False)
        buffer.add_sample(1, 1, 1, False)
        self.assertEqual(buffer._top, 4)
        self.assertSameContent(buffer._valid_transition_indices, [])

    def test_perfect_fit_trajectory(self):
        """
        Index:   0 1 2 3 4 5
        Content: D D E B C C
        Top:           X

        So the only valid transitions should be 0, 4
        """
        buffer = HerReplayBuffer(6, 1, 1)
        for i in range(4):
            buffer.add_sample(1, 1, 1, True)
            buffer.terminate_episode(1, True)
        buffer.add_sample(1, 1, 1, False)
        self.assertEqual(buffer._top, 3)
        self.assertSameContent(buffer._valid_transition_indices, [0, 4])

    def test_goal_state_intervals_are_correct(self):
        """
        Index:   0 1 2 3 4 5 6 7 8 9 10
        Content: A A A A B B B B C C
        Top:                         X
        """
        buffer = HerReplayBuffer(100, 1, 1, num_goals_to_sample=100)
        for _ in range(2):  # A and B
            for _ in range(3):
                buffer.add_sample(1, 1, 1, False)
            buffer.terminate_episode(1, False)

        buffer.add_sample(1, 1, 1, False)  # C
        buffer.add_sample(1, 1, 1, False)  # C
        for i, expected_interval in [
            (0, (1, 4)),
            (1, (2, 4)),
            (2, (3, 4)),
            (3, None),
            (4, (5, 8)),
            (5, (6, 8)),
            (6, (7, 8)),
            (7, None),
            (8, None),
            (9, None),
        ]:
            self.assertEqual(
                buffer._index_to_goal_states_interval[i],
                expected_interval,
            )

    def test_sample_max_num_goals_to_sample(self):
        """
        Index:   0 1 2 3 4 5 6 7 8 9 10
        Content: A A A A B B B B C C
        Top:                         X
        """
        buffer = HerReplayBuffer(100, 1, 1, num_goals_to_sample=2)
        for _ in range(2):  # A and B
            for _ in range(3):
                buffer.add_sample(1, 1, 1, False)
            buffer.terminate_episode(1, False)
        buffer.add_sample(1, 1, 1, False)  # C
        buffer.add_sample(1, 1, 1, False)  # C
        for i, expected_num_goal_states in [
            (0, 2),
            (1, 2),
            (2, 1),
            (3, None),
            (4, 2),
            (5, 2),
            (6, 1),
            (7, None),
            (8, None),
            (9, None),
        ]:
            if expected_num_goal_states is None:
                self.assertEqual(
                    None,
                    buffer._index_to_sampled_goal_states_idxs[i],
                )
            else:
                self.assertEqual(
                    expected_num_goal_states,
                    len(buffer._index_to_sampled_goal_states_idxs[i]),
                )


if __name__ == '__main__':
    unittest.main()
