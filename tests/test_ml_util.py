import unittest

from rlkit.misc.ml_util import StatConditionalSchedule


class TestLossFollowingIntSchedule(unittest.TestCase):

    def test_value_changes_average_1(self):
        schedule = StatConditionalSchedule(
            0,
            (-1, 1),
            1,
        )
        values = []
        for stat in [0, 0, 2, 2, -2]:
            schedule.update(stat)
            values.append(schedule.get_value(0))

        expected = [0, 0, 1, 2, 1]
        self.assertEqual(values, expected)

    def test_value_changes_average_3(self):
        schedule = StatConditionalSchedule(
            0,
            (-1, 1),
            3,
        )
        values = []
        for stat in [0, 0, 2, 2, -2, -2, -2]:
            schedule.update(stat)
            values.append(schedule.get_value(0))

        expected = [0, 0, 0, 1, 1, 1, 0]
        self.assertEqual(values, expected)

    def test_value_changes_average_1_inverse(self):
        schedule = StatConditionalSchedule(
            0,
            (-1, 1),
            1,
            delta=-1,
        )
        values = []
        for stat in [0, 0, 2, 2, -2]:
            schedule.update(stat)
            values.append(schedule.get_value(0))

        expected = [0, 0, -1, -2, -1]
        self.assertEqual(values, expected)

    def test_value_changes_average_3_inverse(self):
        schedule = StatConditionalSchedule(
            0,
            (-1, 1),
            3,
            delta=-1,
        )
        values = []
        for stat in [0, 0, 2, 2, -2, -2, -2]:
            schedule.update(stat)
            values.append(schedule.get_value(0))

        expected = [0, 0, 0, -1, -1, -1, 0]
        self.assertEqual(values, expected)

    def test_value_clipped(self):
        schedule = StatConditionalSchedule(
            0,
            (-1, 1),
            1,
            value_bounds=(-2, 2),
        )
        values = []
        for stat in [2, 2, 2, 2, -2, -2, -2, -2, -2]:
            schedule.update(stat)
            values.append(schedule.get_value(0))

        expected = [1, 2, 2, 2, 1, 0, -1, -2, -2]
        self.assertEqual(values, expected)

    def test_value_clipped_one_way(self):
        schedule = StatConditionalSchedule(
            0,
            (-1, 1),
            1,
            value_bounds=(None, 2),
        )
        values = []
        for stat in [2, 2, 2, 2, -2, -2, -2, -2, -2, -2]:
            schedule.update(stat)
            values.append(schedule.get_value(0))

        expected = [1, 2, 2, 2, 1, 0, -1, -2, -3, -4]
        self.assertEqual(values, expected)

    def test_min_num_stats(self):
        schedule = StatConditionalSchedule(
            0,
            (-1, 1),
            3,
            min_num_stats=0,
        )
        values = []
        for stat in [2, 2, 2, 2]:
            schedule.update(stat)
            values.append(schedule.get_value(0))

        expected = [1, 2, 3, 4]
        self.assertEqual(values, expected)

        schedule = StatConditionalSchedule(
            0,
            (-1, 1),
            3,
            min_num_stats=3,
        )
        values = []
        for stat in [2, 2, 2, 2]:
            schedule.update(stat)
            values.append(schedule.get_value(0))

        expected = [0, 0, 1, 2]
        self.assertEqual(values, expected)

    def test_min_time_between_updates(self):
        schedule = StatConditionalSchedule(
            0,
            (-1, 1),
            3,
            min_num_stats=0,
            min_time_gap_between_value_changes=3,
        )
        values = []
        for t, stat in enumerate([2] * 10):
            schedule.update(stat)
            values.append(schedule.get_value(t))

        expected = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
        self.assertEqual(values, expected)

        schedule = StatConditionalSchedule(
            0,
            (-1, 1),
            3,
            min_num_stats=0,
            min_time_gap_between_value_changes=3,
        )
        values = []
        for t, stat in enumerate([2] * 10):
            values.append(schedule.get_value(t))  # swapped
            schedule.update(stat)

        expected = [0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        self.assertEqual(values, expected)


if __name__ == '__main__':
    unittest.main()