from unittest import TestCase
import numpy as np
import commit
import random
from collections import namedtuple


class Test(TestCase):
    def test_rotate_left(self):
        for i in range(100):
            r = random.randint(0, 2 ** (commit.bit_mask_length - 1))
            self.assertEqual(commit.rotate_left(r), 2 * r)

            rot = r
            for j in range(commit.bit_mask_length):
                rot = commit.rotate_left(rot)
            self.assertEqual(rot, r)

    def test_count_same_bits(self):
        for i in range(100):
            r = random.randint(0, 2 ** commit.bit_mask_length)
            self.assertEqual(commit.count_same_bits(r, r), commit.bit_mask_length)
        for i in range(100):
            r = random.randint(0, 2 ** commit.bit_mask_length)
            s = random.randint(0, 2 ** commit.bit_mask_length)
            rbin = format(r, '0' + str(commit.bit_mask_length) + 'b')
            sbin = format(s, '0' + str(commit.bit_mask_length) + 'b')
            self.assertEqual(commit.count_same_bits(r, s), sum([rbin[i] == sbin[i] for i in range(commit.bit_mask_length)]))

    def test_sim_hash_weighted(self):
        assert (np.all(commit.sim_hash_weighted(commit.all_ones, 5) == np.ones(commit.bit_mask_length)*5))
        assert (np.all(commit.sim_hash_weighted(0, 100) == np.zeros(commit.bit_mask_length) * 100))

        weight = 7
        r = int('10' * (commit.bit_mask_length // 2), 2)
        s = np.zeros(commit.bit_mask_length)
        for i in range(commit.bit_mask_length):
            if i % 2:
                s[i] = -weight
            else:
                s[i] = weight
        assert (np.all(commit.sim_hash_weighted(r, weight) == s))

        weight = 1
        r = int('01' * (commit.bit_mask_length // 2), 2)
        s = np.zeros(commit.bit_mask_length)
        for i in range(1, commit.bit_mask_length):
            if i % 2:
                s[i] = weight
            else:
                s[i] = -weight
        assert (np.all(commit.sim_hash_weighted(r, weight) == s))

    def test_sim_hash_sum_to_bit_mask(self):
        for i in range(100):
            vector = np.zeros(commit.bit_mask_length)
            gz = 0
            for j in range(commit.bit_mask_length):
                vector[j] = random.randint(0, 2 ** commit.bit_mask_length) - random.randint(0, 2 ** commit.bit_mask_length)
                if vector[j] >= 0:
                    gz += 1
            self.assertEqual(commit.count_same_bits(commit.sim_hash_sum_to_bit_mask(vector), commit.all_ones), gz)

    def test_mingle_shingles(self):
        test_list = [("a", 1), ("b", 2), ("c", 3), ("d", 4)]
        self.assertEqual(commit.mingle_shingles(test_list, 1), test_list)
        self.assertEqual(commit.mingle_shingles(test_list, 2), [('ab', 1), ('bc', 2), ('cd', 3)])
        self.assertEqual(commit.mingle_shingles(test_list, 3), [('abc', 2), ('bcd', 3)])
        self.assertEqual(commit.mingle_shingles(test_list, 4), [('abcd', 2)])

        test_list = [("a", 1), ("", 2), ("c", 3), ("d", 4)]
        self.assertEqual(commit.mingle_shingles(test_list, 1), test_list)
        self.assertEqual(commit.mingle_shingles(test_list, 2), [('a', 1), ('c', 2), ('cd', 3)])
        self.assertEqual(commit.mingle_shingles(test_list, 3), [('ac', 2), ('cd', 3)])
        self.assertEqual(commit.mingle_shingles(test_list, 4), [('acd', 2)])

    def test_get_hunk_strings(self):
        Hunk_line = namedtuple('line', ['value', 'is_context', 'is_added', 'is_removed'])
        hunk = [Hunk_line(value=str(i), is_context=i % 3 == 0, is_added=i % 3 == 1, is_removed=i % 3 == 2) for i in range(10)]
        w_context, w_body = 1, 10
        self.assertEqual(commit.get_hunk_strings(hunk, w_context, w_body), [('0', 1), ('+1', 10), ('-2', 10), ('3', 1), ('+4', 10), ('-5', 10), ('6', 1), ('+7', 10), ('-8', 10), ('9', 1)])