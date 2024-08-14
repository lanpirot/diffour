import unidiff
import re
import numpy as np
import mmh3

bit_mask_length = 64
cherry_commit_message_pattern = r"\(cherry picked from commit ([a-fA-F0-9]{40})\)"
git_origin_pattern = r"GitOrigin-RevId: ([a-fA-F0-9]{40})"
date_id = 0  #the cherry picks are sorted by date, we give them an ID by our processing order


def get_date_id():
    global date_id
    date_id += 1
    return date_id


def rotate_left(bitmask):
    return ((bitmask << 1) & ((1 << bit_mask_length) - 1)) | (bitmask >> (bit_mask_length - 1))


def count_same_bits(num1, num2):
    same_bits = ~(num1 ^ num2)
    count = bin(same_bits & ((1 << max(num1.bit_length(), num2.bit_length())) - 1)).count('1')
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


def right_strip(wdiff):
    stripped = []
    for wd in wdiff:
        if wd[0] and wd[0][-1] == "\n":
            wd = wd[0][:-1], wd[1]
        stripped.append((wd[0].rstrip(), wd[1]))
    return stripped


#single lines are no great shingles, connect them to give each other context
def mingle_shingles(wdiff, n):
    return [("".join([wdiff[j][0] for j in range(i, i + n)]), sum([wdiff[j][1] for j in range(i, i + n)]) // n) for i in
            range(len(wdiff) - n + 1)]


class Commit:
    def __init__(self, commit_str, diff_marker):
        self.parent_id = None
        self.is_root = None
        self.commit_id = None
        self.rev_id = None
        self.claimed_cherry = None
        self.author = None
        self.commit_message = None
        self.parseable = True
        self.patch_set = None
        self.bit_mask = None
        self.date = get_date_id()
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
        self.claimed_cherry = self.get_claimed_cherry()

    def has_rev_id(self):
        return re.search(git_origin_pattern, self.commit_message)

    def get_rev_id(self):
        if not self.has_rev_id():
            return None
        return re.search(git_origin_pattern, self.commit_message).group(1)

    # does the commit message claim it was a cherry pick?
    def claims_cherry_pick(self):
        return re.search(cherry_commit_message_pattern, self.commit_message)

    def get_claimed_cherry(self):
        if not self.claims_cherry_pick():
            return None
        match = re.search(cherry_commit_message_pattern, self.commit_message)
        return match.group(1)

    def other_is_my_cherry(self, other_commit):
        if self.claimed_cherry:
            return self.claimed_cherry == other_commit.commit_id or self.claimed_cherry == other_commit.rev_id
        return False

    def get_ordered_commit_pair(self, other_commit):
        if self.date > other_commit.date:
            return self, other_commit
        return other_commit, self

    # a list of weighted strings, the strings are mostly the diff itself
    # <string>, <weight>
    # commit_message, 1
    # file_before, file_after 10
    # hunk_header 1
    # context_before 1
    # body, 10
    # context_after 1
    def get_weighted_diff(self):
        w_message = 1
        w_filename = 10
        w_hheader = 1
        w_context = 1
        w_body = 10

        weighted_diff = []
        weighted_diff += [(self.commit_message, w_message)]
        if self.parseable:
            for patch in self.patch_set:
                weighted_diff += [(patch.source_file, w_filename), (patch.target_file, w_filename)]
                for hunk in patch:
                    header = [(",".join([str(i) for i in [hunk.source_start, hunk.source_length, hunk.target_start,
                                                          hunk.target_length]]), w_hheader)]
                    body = right_strip(
                        [(line.value, w_context) if line.is_context else (line.value, w_body) for line in hunk])
                    weighted_diff += header
                    weighted_diff += body
                    weighted_diff += mingle_shingles(body, 2)
        return weighted_diff

    # create a bit_mask (a signature) of a commit
    def get_bit_mask(self):
        wdiff = self.get_weighted_diff()
        #wdiff = right_strip(wdiff)

        sim_hash_sum = np.zeros(bit_mask_length)
        for (line, weight) in wdiff:
            #hsh = hash(line)
            mask = (1 << bit_mask_length) - 1
            hsh = mmh3.hash128(line) & mask

            sim_hash_sum += sim_hash_weighted(hsh, weight)
        return sim_hash_sum_to_bit_mask(sim_hash_sum)
