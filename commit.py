import unidiff
import re
import numpy as np
import mmh3


bit_mask_length = 64
all_ones = 2**bit_mask_length - 1
cherry_commit_message_pattern = r"\(cherry picked from commit ([a-fA-F0-9]{40})\)"
git_origin_pattern = r"GitOrigin-RevId: ([a-fA-F0-9]{40})"



def rotate_left(bitmask):
    return ((bitmask << 1) & all_ones) | (bitmask >> (bit_mask_length - 1))


def count_same_bits(num1, num2):
    same_bits = ~(num1 ^ num2)
    count = bin(same_bits & all_ones).count('1')
    return count


def sim_hash_weighted(hsh, weight):
    sim_hash_weighted = np.zeros(bit_mask_length)
    digit = bit_mask_length - 1
    while hsh and digit >= 0:
        if hsh & 1:
            sim_hash_weighted[digit] = weight
        else:
            sim_hash_weighted[digit] = -weight
        hsh >>= 1
        digit -= 1
    return sim_hash_weighted


def sim_hash_sum_to_bit_mask(sim_hash_sum):
    bm = 0
    for digit in range(bit_mask_length):
        bm <<= 1
        if sim_hash_sum[digit] >= 0:
            bm += 1
    return bm


#single lines are no great shingles, connect them to give each other context
def mingle_shingles(wdiff, n):
    return [("".join([wdiff[j][0] for j in range(i, i + n)]), sum([wdiff[j][1] for j in range(i, i + n)]) // n) for i in range(len(wdiff) - n + 1)]

#the unidiff lib cuts off the leading - and + of the hunk body
#add it back in to not confuse "- print()" with "+ print()"
def get_hunk_strings(hunk, w_context, w_body, rs):
    ret = []
    for line in hunk:
        l = line.value
        if rs:
            l = l.rstrip()
        if line.is_context:
            ret.append((l, w_context))
        elif line.is_added:
            ret.append(("+"+l, w_body))
        elif line.is_removed:
            ret.append(("-"+l, w_body))
        elif line.value == ' No newline at end of file\n':
            ret.append((l, w_context))
        else:
            raise
    return ret

class Commit:
    date_id = 2 ** bit_mask_length  # the cherry picks are sorted by date, we give them an ID by our processing order
    normal_weight = 1
    special_weight = 10
    rs = True

    def __init__(self, commit_str, diff_marker):
        self.date = self.__class__.date_id
        self.__class__.date_id -= 1

        self.parent_id = None
        self.is_root = None
        self.commit_id = None
        self.rev_id = None
        self.claimed_cherries = None
        self.author = None
        self.commit_message = None
        self.parseable = True
        self.patch_set = None
        self.bit_mask = None
        self.parse_commit_str(commit_str, diff_marker)

    #ugly parser of our git string
    def parse_commit_str(self, commit_str, diff_marker):
        commit_str = commit_str.split(diff_marker + "\n")
        if len(commit_str) != 2:
            if commit_str[0].splitlines()[0]:
                raise
            #create dummy commit string for root
            commit_str.append('')
        start = commit_str[0].split("\n", 3)

        if len(start) < 4:
            raise
        (parent_id, commit_id, author, commit_message) = (start[0], start[1], start[2], start[3:][0])

        commit_diff = commit_str[1]
        if len(commit_diff) > 1 and commit_diff[-2] == commit_diff[-1] == "\n":
            commit_diff = commit_diff[:-1]
        try:
            patch_set = unidiff.PatchSet(commit_diff)
        except unidiff.UnidiffParseError as e:
            self.parseable = False
            patch_set = None
        self.init(parent_id, commit_id, author, commit_message, patch_set)

    def init(self, parent_id, commit_id, author, commit_message, patch_set):
        self.parent_id = parent_id
        self.is_root = len(self.parent_id) == 0
        self.commit_id = commit_id
        self.author = author
        self.commit_message = commit_message
        self.patch_set = patch_set
        self.bit_mask = self.get_bit_mask()
        self.rev_id = self.get_rev_id()
        self.claimed_cherries = self.get_claimed_cherries()

    def has_rev_id(self):
        return re.search(git_origin_pattern, self.commit_message) is not None

    def get_rev_id(self):
        if not self.has_rev_id():
            return None
        return re.search(git_origin_pattern, self.commit_message).group(1)

    # does the commit message claim it was a cherry pick?
    def claims_cherry_pick(self):
        return re.search(cherry_commit_message_pattern, self.commit_message) is not None

    def get_claimed_cherries(self):
        if not self.claims_cherry_pick():
            return []
        matches = re.findall(cherry_commit_message_pattern, self.commit_message)
        return list(matches)

    def other_is_my_cherry(self, other_commit):
        if self.claimed_cherries:
            return other_commit.commit_id in self.claimed_cherries or other_commit.rev_id in self.claimed_cherries
        return False

    def get_ordered_commit_pair(self, other_commit):
        if self.date < other_commit.date:
            return self, other_commit
        return other_commit, self

    # a list of weighted strings, the strings are mostly the diff itself
    # <string>, <weight>
    def get_weighted_diff(self):
        w_message = self.__class__.normal_weight
        w_filename = self.__class__.special_weight
        w_hheader = self.__class__.normal_weight
        w_context = self.__class__.normal_weight
        w_body = self.__class__.special_weight

        if not self.parseable:
            return None
        weighted_diff = []
        weighted_diff += [(self.commit_message, w_message)]

        for patch in self.patch_set:
            weighted_diff += [(patch.source_file, w_filename), (patch.target_file, w_filename)]
            for hunk in patch:
                header = [(",".join([str(i) for i in [hunk.source_start, hunk.source_length, hunk.target_start, hunk.target_length]]), w_hheader)]
                body = get_hunk_strings(hunk, w_context, w_body, self.__class__.rs)
                weighted_diff += header
                weighted_diff += body
                weighted_diff += mingle_shingles(body, 2)
        return weighted_diff

    # create a bit_mask (a signature) of a commit
    def get_bit_mask(self):
        wdiff = self.get_weighted_diff()

        sim_hash_sum = np.zeros(bit_mask_length)
        for (line, weight) in wdiff:
            mask = (1 << bit_mask_length) - 1
            hsh = mmh3.hash128(line) & mask

            sim_hash_sum += sim_hash_weighted(hsh, weight)
        return sim_hash_sum_to_bit_mask(sim_hash_sum)
