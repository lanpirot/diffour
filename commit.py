import textdistance
import unidiff
import re
import numpy as np
import mmh3
from dataclasses import dataclass

bit_mask_length: int = 64
all_ones: int = 2 ** bit_mask_length - 1
git_hash40: str = "[a-fA-F0-9]{40}"
cherry_commit_message_pattern: str = rf"\(cherry picked from commit ({git_hash40})\)|X-original-commit: ({git_hash40})"
git_origin_pattern: str = rf"GitOrigin-RevId: ({git_hash40})"
max_levenshtein_string_length: int = 10 ** 4

min_bit_similarity: float = 0.9
min_levenshtein_similarity: float = 0.7


def is_similar_bitmask(bitmask1, bitmask2):
    if not bitmask1 or not bitmask2:
        return False, None
    same_bits = count_same_bits(bitmask1, bitmask2)
    bit_sim = same_bits / bit_mask_length
    return bit_sim >= min_bit_similarity, bit_sim


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


# single lines are no great shingles, connect them to give each other context
def mingle_shingles(wdiff, n):
    return [("".join([wdiff[j][0] for j in range(i, i + n)]), sum([wdiff[j][1] for j in range(i, i + n)]) // n) for i in range(len(wdiff) - n + 1)]


# the unidiff lib cuts off the leading - and + of the hunk body
# add it back in to not confuse "- print()" with "+ print()"
def get_hunk_strings(hunk, w_context, w_body, rs):
    ret = []
    for line in hunk:
        l = line.value
        if rs:
            l = l.rstrip()
        if line.is_context:
            ret.append((l, w_context))
        elif line.is_added:
            ret.append(("+" + l, w_body))
        elif line.is_removed:
            ret.append(("-" + l, w_body))
        elif line.value == ' No newline at end of file\n':
            ret.append((l, w_context))
        else:
            raise unidiff.UnidiffParseError
    return ret


# create a dummy cherry commit to populate the git graph with unsampled, but known cherries
def dummy_cherry_commit(commit_id, diff_marker):
    dummy = Commit(f"\n{commit_id}\nA. Nonymous\n!!Dummy Commit!!\n{diff_marker}\n", diff_marker)
    return dummy


class Commit:
    date_id: int = 2 ** bit_mask_length  # the cherry picks are sorted by date, we give them an ID by our processing order
    normal_weight: int = 1
    special_weight: int = 10
    rs: bool = True

    def __init__(self, commit_str, diff_marker):
        self.date = self.__class__.date_id
        self.__class__.date_id -= 1

        (parent_id, commit_id, author, commit_message, patch_set, parseable) = self.parse_commit_str(commit_str, diff_marker)

        self.commit_message: str = commit_message
        self.author: str = author
        self.parent_id: str = parent_id
        self.commit_id: str = commit_id
        self.is_root: bool = len(self.parent_id) == 0
        self.rev_id: str = self.get_rev_id()
        self.explicit_cherries: list = []
        self.patch_set: unidiff.PatchSet = patch_set
        self.parseable: bool = parseable
        self.bit_mask: int
        if self.parseable:
            self.bit_mask = self.get_bit_mask()
        else:
            self.bit_mask = None
        self.neighbor_connections: list = []

    # ugly parser of our git string
    def parse_commit_str(self, commit_str, diff_marker):
        commit_str = commit_str.split(diff_marker + "\n")
        if len(commit_str) != 2:
            raise ValueError
        commit_header = commit_str[0].split("\n", 3)

        if len(commit_header) < 4:
            raise ValueError
        (parent_id, commit_id, author, commit_message) = (commit_header[0], commit_header[1], commit_header[2], commit_header[3:][0])

        commit_diff = commit_str[1]
        if len(commit_diff) > 1 and commit_diff[-2] == commit_diff[-1] == "\n":
            commit_diff = commit_diff[:-1]
        try:
            patch_set = unidiff.PatchSet(commit_diff)
            if len(patch_set) == 0:
                raise unidiff.UnidiffParseError
            parseable = True
        except unidiff.UnidiffParseError as e:
            parseable = False
            patch_set = None
        return parent_id, commit_id, author, commit_message, patch_set, parseable

    # we remove the index line of a patch-diff (it states the hashsum of a file, which is okay to differ)
    # we also limit the maximal string length to avoid system crashes, and get some speed
    # we heuristically search the start and end only for long patches
    def clean_patch_string(self):
        patch_string = self.patch_set.__str__()
        if len(patch_string) > max_levenshtein_string_length:
            patch_string = patch_string[:max_levenshtein_string_length // 2] + patch_string[-max_levenshtein_string_length // 2:]
        return '\n'.join(line for line in patch_string.splitlines() if not line.startswith("index "))

    def has_similar_text_to(self, neighbor):
        if not self.parseable or not neighbor.parseable:
            return False, None
        if len(self.patch_set.__str__()) * 2 < len(neighbor.patch_set.__str__()) or len(self.patch_set.__str__()) > len(
                neighbor.patch_set.__str__()) * 2:
            return False, 0
        patch_string1, patch_string2 = self.clean_patch_string(), neighbor.clean_patch_string()
        similarity = textdistance.levenshtein.normalized_similarity(patch_string1, patch_string2)
        return similarity >= min_levenshtein_similarity, similarity

    def add_neighbor(self, neighbor_commit):
        # we don't need to add a neighbor twice
        if neighbor_commit.commit_id in [c.neighbor.commit_id for c in self.neighbor_connections]:
            return

        bit_sim, bit_sim_level = is_similar_bitmask(self.bit_mask, neighbor_commit.bit_mask)
        text_sim, levenshtein_sim_level = self.has_similar_text_to(neighbor_commit)
        sim = bit_sim and text_sim
        explicit_cherrypick = self.other_is_in_my_cherries(neighbor_commit) or neighbor_commit.other_is_in_my_cherries(self)

        if sim or explicit_cherrypick:
            neighbor = Neighbor(neighbor=neighbor_commit, sim=sim, bit_sim=bit_sim_level, levenshtein_sim=levenshtein_sim_level,
                                explicit_cherrypick=explicit_cherrypick)
            self.neighbor_connections.append(neighbor)

    def has_rev_id(self):
        return re.search(git_origin_pattern, self.commit_message) is not None

    def get_rev_id(self):
        if not self.has_rev_id():
            return None
        return re.search(git_origin_pattern, self.commit_message).group(1)

    # does the commit message claim it has a cherrypick?
    def has_explicit_cherrypick(self):
        return re.search(cherry_commit_message_pattern, self.commit_message) is not None

    def get_explicit_cherrypicks(self):
        if not self.has_explicit_cherrypick():
            return []
        matches = re.findall(cherry_commit_message_pattern, self.commit_message)
        flattened_matches = [value for t in matches for value in t if value]
        return flattened_matches

    def other_is_in_my_cherries(self, other_commit):
        return other_commit.commit_id in self.explicit_cherries or other_commit.rev_id in self.explicit_cherries

    def is_younger_than(self, other_commit):
        return self.date > other_commit.date

    # the older commit first, the younger second
    def get_ordered_commit_pair(self, other_commit):
        if self.is_younger_than(other_commit):
            return other_commit, self
        return self, other_commit

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
        # weighted_diff += [(self.commit_message, w_message)]

        for patch in self.patch_set:
            weighted_diff += [(patch.source_file, w_filename), (patch.target_file, w_filename)]
            if patch.is_binary_file:
                # Binary file detected, we add the commit message, to compensate for us not knowing the actual file diff.
                # This means, we hope the commit message gives us clues, what is going on in the binary files.
                for l in self.commit_message.splitlines():
                    weighted_diff += [(l, w_message)]
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


@dataclass
class Neighbor:
    neighbor: Commit
    sim: bool
    bit_sim: float
    levenshtein_sim: float
    explicit_cherrypick: bool
