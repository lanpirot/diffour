from typing import Optional

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

max_bit_diff: int = 4
min_bit_similarity: float = (bit_mask_length - max_bit_diff) / bit_mask_length
min_levenshtein_similarity: float = 0.75


# computes the similarity (and judges it against our cutoff "min_bit_similarity") of two bitmasks
def is_similar_bitmask(bitmask1: int, bitmask2: int) -> tuple[bool, Optional[float]]:
    if bitmask1 is None or bitmask2 is None:
        return False, None
    same_bits: int = count_same_bits(bitmask1, bitmask2)
    bit_sim: float = same_bits / bit_mask_length
    return bit_sim >= min_bit_similarity, bit_sim


# left rotation of a bitmask, where the first bit gets rotated to the last place
def rotate_left(bitmask: int) -> int:
    return ((bitmask << 1) & all_ones) | (bitmask >> (bit_mask_length - 1))


def count_same_bits(num1: int, num2: int) -> int:
    same_bits: int = ~(num1 ^ num2)
    count: int = bin(same_bits & all_ones).count('1')
    return count


# given a hash (bitmask), construct a vector:
# put +weight for all set bits of bitmask, put -weight for all unset bits of bitmask
# into the vector at the position of the bit, we analyze
def sim_hash_weighted(hsh: int, weight: int) -> np.ndarray:
    sim_hash_weight: np.ndarray = np.zeros(bit_mask_length, dtype=int)
    digit: int = bit_mask_length - 1
    while hsh and digit >= 0:
        if hsh & 1:
            sim_hash_weight[digit] = weight
        else:
            sim_hash_weight[digit] = -weight
        hsh >>= 1
        digit -= 1
    return sim_hash_weight


# translate the vector from the function-calls of "sim_hash_weighted" back to a bitmask
# in our vector: all numbers > 0 -> set bit, all numbers < 0 -> unset bit
def sim_hash_sum_to_bit_mask(sim_hash_sum: np.ndarray) -> int:
    bm: int = 0
    for digit in range(bit_mask_length):
        bm <<= 1
        if sim_hash_sum[digit] >= 0:
            bm += 1
    return bm


# single lines are no great shingles, connect them to give each other context
def mingle_shingles(wdiff: list[tuple[str, int]], n: int) -> list[tuple[str, int]]:
    return [("".join([wdiff[j][0] for j in range(i, i + n)]), sum([wdiff[j][1] for j in range(i, i + n)]) // n) for i in range(len(wdiff) - n + 1)]


# the unidiff lib cuts off the leading - and + of the hunk body
# add it back in to not confuse "- print()" with "+ print()" lines
def get_hunk_strings(hunk: unidiff.Hunk, w_context: int, w_body: int, rs: bool) -> list[tuple[str, int]]:
    ret = []
    for hunk_line in hunk:
        line = hunk_line.value
        if rs:
            line = line.rstrip()
        if hunk_line.is_context:
            ret.append((line, w_context))
        elif hunk_line.is_added:
            ret.append(("+" + line, w_body))
        elif hunk_line.is_removed:
            ret.append(("-" + line, w_body))
        elif hunk_line.line_type == "\\" or hunk_line.line_type == "":
            ret.append((line, w_context))
        else:
            raise unidiff.UnidiffParseError
    return ret


# create a dummy cherry commit to populate the git graph with cherries we did not sample, but know of
def dummy_cherry_commit(commit_id: str, diff_marker: str) -> 'Commit':
    return Commit(f"\n{commit_id}\nA. Nonymous\n!!Dummy Commit!!\n{diff_marker}\n", diff_marker)


# Commit: a class to store all information of a commit, we can gather from the git log parsing
#         we also compute a bitmask-signature of its udiff (and commit message for binary files) for SimHash
#         we only store parts of long udiff patches, see max_levenshtein_string_length,
#         1. to limit memory_impact of giant commits
#         2. to limit time comparisons of long udiff patches
class Commit:
    date_id: int = 2 ** bit_mask_length  # the cherrypicks are sorted by date, we give them an ID by our processing order
    normal_weight: int = 1
    special_weight: int = 10
    rs: bool = True

    def __init__(self, commit_str: str, diff_marker: str) -> None:
        self.date: int = self.__class__.date_id
        self.__class__.date_id -= 1

        (parent_ids, commit_id, author, commit_message, patch_string, parseable, bit_mask) = self.parse_commit_str(commit_str, diff_marker)

        self.commit_message: str = commit_message
        self.author: str = author
        self.parent_ids: list[str] = parent_ids
        self.commit_id: str = commit_id
        self.is_root: bool = len(self.parent_ids) == 0
        self.rev_id: Optional[str] = self.get_rev_id()
        self.explicit_cherries: list = self.get_explicit_cherrypicks()
        self.patch_string: str = patch_string
        self.parseable: bool = parseable
        self.bit_mask: int = bit_mask
        self.neighbor_connections: list = []

    # an ugly parser of our git string, expects a string of the form
    # parent_id
    # commit_id
    # author_name
    # commit_message
    # <DIFF_MARKER>
    # udiff
    def parse_commit_str(self, commit_str: str, diff_marker: str):
        commit_str: list[str] = commit_str.split(diff_marker + "\n")
        if len(commit_str) != 2:
            commit_str = commit_str[0].split(diff_marker)
            if len(commit_str) != 2:
                raise ValueError
        commit_header: list[str] = commit_str[0].split("\n", 3)

        if len(commit_header) < 4:
            raise ValueError
        (parent_idstring, commit_id, author, commit_message) = (commit_header[0], commit_header[1], commit_header[2], commit_header[3:][0])
        parent_ids: list[str] = parent_idstring.split(" ")
        parent_ids = [p for p in parent_ids if len(p) > 0]

        commit_diff: str = commit_str[1]
        if len(commit_diff) > 1 and commit_diff[-2] == commit_diff[-1] == "\n":
            commit_diff = commit_diff[:-1]

        patch_string: Optional[str]
        bit_mask: Optional[int]
        parseable: bool
        try:
            patch_set: unidiff.PatchSet = unidiff.PatchSet(commit_diff)
            if len(patch_set) == 0:
                raise unidiff.UnidiffParseError
            # we compute the bitmask here, so we can forget the whole patch_string right away
            bit_mask = self.get_bit_mask(patch_set, commit_message)
            patch_string = self.clean_patch_string(patch_set.__str__())
            parseable = True
        except unidiff.UnidiffParseError:
            parseable = False
            bit_mask = None
            patch_string = None
        return parent_ids, commit_id, author, commit_message, patch_string, parseable, bit_mask

    # we remove the index line of a patch-diff (it states the hashsum of a file, which is okay to differ)
    # we also limit the maximal string length to avoid system crashes, and get some speed
    # we heuristically search the start and end only for long patches
    @staticmethod
    def clean_patch_string(patch_string: str) -> str:
        if len(patch_string) > max_levenshtein_string_length:
            patch_string = patch_string[:max_levenshtein_string_length // 2] + patch_string[-max_levenshtein_string_length // 2:]
        return '\n'.join(line for line in patch_string.splitlines() if not line.startswith("index "))

    # we use Levenshtein-Similarity to compute the similarity of two texts
    # we also judge it against a cutoff "min_levenshtein_similarity"
    def has_similar_text_to(self, neighbor: 'Commit') -> tuple[bool, Optional[float]]:
        if not self.parseable or not neighbor.parseable:
            return False, None
        if len(self.patch_string) * 2 < len(neighbor.patch_string) or len(self.patch_string) > len(neighbor.patch_string) * 2:
            return False, 0
        # TODO: Match first half first, see if high score is achieved, only then match second half
        similarity: float = textdistance.levenshtein.normalized_similarity(self.patch_string, neighbor.patch_string)
        return similarity >= min_levenshtein_similarity, similarity

    def already_neighbors(self, other):
        return other.commit_id in [c.neighbor.commit_id for c in self.neighbor_connections] or self.commit_id in [c.neighbor.commit_id for c in
                                                                                                                  other.neighbor_connections]

    # add a neighbor edge for our neighbor graph
    # we expect edges of type: strong similarity (bitwise, levenshtein), explicit cherrypick, git-parent-relation
    def add_neighbor(self, other: 'Commit', connect_children) -> None:
        # we don't need to add a neighbor twice
        if self.already_neighbors(other):
            return

        is_child_of = self.is_child_of(other)
        if is_child_of:
            bit_sim, bit_sim_level = False, None
            text_sim, levenshtein_sim_level = False, None
        else:
            bit_sim, bit_sim_level = is_similar_bitmask(self.bit_mask, other.bit_mask)
            text_sim, levenshtein_sim_level = self.has_similar_text_to(other)
        sim = bit_sim and text_sim
        explicit_cherrypick = self.other_is_in_my_cherries(other)

        if sim or explicit_cherrypick or (is_child_of and connect_children):
            neighbor: Neighbor = Neighbor(neighbor=other, sim=sim, bit_sim=bit_sim_level, levenshtein_sim=levenshtein_sim_level,
                                          explicit_cherrypick=explicit_cherrypick, is_child_of=is_child_of)
            self.neighbor_connections.append(neighbor)

    def is_child_of(self, other_commit: 'Commit') -> bool:
        return other_commit.commit_id in self.parent_ids

    # test if commit message features a "GitOrigin-RevId"
    def has_rev_id(self) -> bool:
        return re.search(git_origin_pattern, self.commit_message) is not None

    def get_rev_id(self) -> Optional[str]:
        if not self.has_rev_id():
            return None
        return re.search(git_origin_pattern, self.commit_message).group(1)

    def has_explicit_cherrypick(self):
        return re.search(cherry_commit_message_pattern, self.commit_message) is not None

    def get_explicit_cherrypicks(self) -> list[str]:
        if not self.has_explicit_cherrypick():
            return []
        matches: list = re.findall(cherry_commit_message_pattern, self.commit_message)
        flattened_matches: list[str] = [value for t in matches for value in t if value]
        return flattened_matches

    def other_is_in_my_cherries(self, other_commit: 'Commit') -> bool:
        return other_commit.commit_id in self.explicit_cherries

    # git log provides an ordering of commits, we parse commits in that order
    # was self parsed before other_commit?
    def is_younger_than(self, other_commit: 'Commit') -> bool:
        return self.date > other_commit.date

    # the older commit first, the younger second
    def get_ordered_commit_pair(self, other_commit: 'Commit') -> tuple['Commit', 'Commit']:
        if self.is_younger_than(other_commit):
            return other_commit, self
        return self, other_commit

    # a list of weighted strings, the strings are mostly the diff itself
    # <string>, <weight>
    def get_weighted_diff(self, patch_set: unidiff.PatchSet, commit_message: str) -> list[tuple[str, int]]:
        w_message: int = self.__class__.normal_weight
        w_filename: int = self.__class__.special_weight
        w_header: int = self.__class__.normal_weight
        w_context: int = self.__class__.normal_weight
        w_body: int = self.__class__.special_weight

        weighted_diff: list[tuple[str, int]] = []

        for patch in patch_set:
            weighted_diff += [(patch.source_file, w_filename), (patch.target_file, w_filename)]
            if patch.is_binary_file:
                # Binary file detected, we add the commit message, to compensate for us not knowing the actual file diff.
                # This means, we hope the commit message gives us clues, what is going on in the binary files.
                for line in commit_message.splitlines():
                    weighted_diff += [(line, w_message)]
            for hunk in patch:
                header: tuple[str, int] = (",".join([str(i) for i in [hunk.source_start, hunk.source_length, hunk.target_start, hunk.target_length]]), w_header)
                body: list[tuple[str, int]] = get_hunk_strings(hunk, w_context, w_body, self.__class__.rs)
                weighted_diff = [header] + body
                weighted_diff += mingle_shingles(body, 2)
        return weighted_diff

    # create a bit_mask (a signature) of a commit
    def get_bit_mask(self, patch_set: unidiff.PatchSet, commit_message: str) -> int:
        wdiff: list[tuple[str, int]] = self.get_weighted_diff(patch_set, commit_message)

        sim_hash_sum: np.ndarray = np.zeros(bit_mask_length, dtype=int)
        for (line, weight) in wdiff:
            mask: int = (1 << bit_mask_length) - 1
            hsh: int = mmh3.hash128(line) & mask

            sim_hash_sum += sim_hash_weighted(hsh, weight)
        return sim_hash_sum_to_bit_mask(sim_hash_sum)


# Neighbor: a class for the edges of our Neighbor-Graph
# used for the different edge types and edge weights
@dataclass
class Neighbor:
    neighbor: Commit
    sim: bool
    bit_sim: float
    levenshtein_sim: float
    explicit_cherrypick: bool
    is_child_of: bool
