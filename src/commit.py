from typing import Optional

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

max_bit_diff: int = 5
min_bit_similarity: float = (bit_mask_length - max_bit_diff) / bit_mask_length
min_patch_similarity = 0.5


# computes the similarity (and judges it against our cutoff "min_bit_similarity") of two signatures
def is_similar_signature(sign1: int, sign2: int) -> tuple[bool, Optional[float]]:
    if sign1 is None or sign2 is None:
        return False, None
    same_bits: int = count_same_bits(sign1, sign2)
    bit_sim: float = same_bits / bit_mask_length
    return bit_sim >= min_bit_similarity, bit_sim


# left rotation of a signature, where the first bit gets rotated to the last place
def rotate_left(sign: int) -> int:
    return ((sign << 1) & all_ones) | (sign >> (bit_mask_length - 1))


def count_same_bits(num1: int, num2: int) -> int:
    same_bits: int = ~(num1 ^ num2)
    count: int = bin(same_bits & all_ones).count('1')
    return count


# given a hash (signature), construct a vector:
# put +weight for all set bits of signature, put -weight for all unset bits of signature
# into the vector at the position of the bit, we analyze
def sim_hash_weighted(hsh: int, weight: int = 1) -> np.ndarray:
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


# translate the vector from the function-calls of "sim_hash_weighted" back to a signature
# in our vector: all numbers > 0 -> set bit, all numbers < 0 -> unset bit
def sim_hash_sum_to_signature(sim_hash_sum: np.ndarray) -> int:
    bm: int = 0
    for digit in range(bit_mask_length):
        bm <<= 1
        if sim_hash_sum[digit] >= 0:
            bm += 1
    return bm


def sign_commit_rough(patch_set: set[int]) -> int:
    sim_hash_sum: np.ndarray = np.zeros(bit_mask_length, dtype=int)
    for patch in patch_set:
        sim_hash_sum += sim_hash_weighted(patch)
    return sim_hash_sum_to_signature(sim_hash_sum)


def sign_commit_fine(patch_str: str) -> int:
    return mmh3.hash64('\n'.join(line for line in patch_str.splitlines() if not line.startswith("index ")))[0]


def sign_hunk(hunk_str: str) -> int:
    return mmh3.hash64(hunk_str)[0]


# create a dummy cherry commit to populate the git graph with cherries we did not sample, but know of
def dummy_cherry_commit(commit_id: str, diff_marker: str) -> 'Commit':
    return Commit(f"\n{commit_id}\nA. Nonymous\n!!Dummy Commit!!\n{diff_marker}\n", diff_marker)


def get_hunk_string(hunk: unidiff.patch) -> str:
    # ret = ",".join([str(i) for i in [hunk.source_start, hunk.source_length, hunk.target_start, hunk.target_length]]) + "\n"
    ret = ""
    for hunk_line in hunk:
        if hunk_line.is_context:
            ret += hunk_line.value
        elif hunk_line.is_added:
            ret += "+" + hunk_line.value
        elif hunk_line.is_removed:
            ret += "-" + hunk_line.value
        else:
            continue
    return ret


# an ugly parser of our git string, expects a string of the form
# parent_id
# commit_id
# author_name
# commit_message
# <DIFF_MARKER>
# udiff
def parse_commit_str(commit_str: str, diff_marker: str):
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

    patch_set: Optional[unidiff.PatchSet]
    parseable: bool
    try:
        patch_set: unidiff.PatchSet = unidiff.PatchSet(commit_diff)
        parseable = True
        if len(patch_set) == 0:
            raise unidiff.UnidiffParseError
    except unidiff.UnidiffParseError:
        parseable = False
        patch_set = None
    return parent_ids, commit_id, author, commit_message, patch_set, parseable


# Commit: a class to store all information of a commit, we can gather from the git log parsing
#         we also compute a rough, but locality sensitive signature of its udiff (and commit message for binary files) for SimHash
#         and a fine hash of the whole diff
class Commit:
    _date_id: int = 2 ** bit_mask_length  # the cherrypicks are sorted by date, we give them an ID by our processing order

    def __init__(self, commit_str: str, diff_marker: str) -> None:
        self.date: int = self.__class__._date_id
        self.__class__._date_id -= 1

        (parent_ids, commit_id, author, commit_message, patch_set, parseable) = parse_commit_str(commit_str, diff_marker)
        # TODO: find out branch of commit

        self.commit_message: str = commit_message
        self.author: str = author
        self.parent_ids: list[str] = parent_ids
        self.commit_id: str = commit_id
        self.is_root: bool = len(self.parent_ids) == 0
        self.rev_id: Optional[str] = self.get_rev_id()
        self.has_explicit_cherries: bool = self.has_explicit_cherrypick()
        self.explicit_cherries: list = self.get_explicit_cherrypicks()
        self.parseable: bool = parseable
        if self.parseable:
            self.patch_set: set[int] = self.signature_patch_set(patch_set)
            self.commit_lsh_signature: int = sign_commit_rough(self.patch_set)
            self.commit_fine_signature: int = sign_commit_fine(patch_set.__str__())
        else:
            self.patch_set, self.commit_lsh_signature, self.commit_fine_signature = None, None, None
        self.neighbor_connections: set[Neighbor] = set()

    # we use Jaccard-Similarity of Hunks of a Diff
    def is_similar_patch_to(self, neighbor: 'Commit') -> tuple[bool, Optional[float]]:
        ps1, ps2 = self.patch_set, neighbor.patch_set
        if not ps1 or not ps2:
            return False, None
        similarity = len(ps1.intersection(ps2)) / len(ps1.union(ps2))
        return similarity >= min_patch_similarity, similarity

    def already_neighbors(self, other):
        return other in self.neighbor_connections or self in other.neighbor_connections or self == other

    # add a neighbor edge for our neighbor graph
    # we expect edges of type: strong similarity (bitwise, patch_sim), explicit cherrypick, git-parent-relation
    def add_neighbor(self, other: 'Commit', patch_sim_given: tuple[bool, float] = None) -> None:
        # we don't need to add a neighbor twice
        if self.already_neighbors(other):
            return
        neighbor: Neighbor

        if self.is_child_of(other):
            neighbor = Neighbor(neighbor=other, sim=False, bit_sim=0, patch_sim=0, explicit_cherrypick=False, is_child_of=True)
        else:
            if patch_sim_given:
                patch_sim, patch_sim_level = patch_sim_given
            else:
                patch_sim, patch_sim_level = self.is_similar_patch_to(other)
            bit_sim, bit_sim_level = is_similar_signature(self.commit_lsh_signature, other.commit_lsh_signature)
            is_similar = bit_sim and patch_sim

            if other.other_is_in_my_cherries(self):
                neighbor = Neighbor(neighbor=self, sim=is_similar, bit_sim=bit_sim_level, patch_sim=patch_sim_level, explicit_cherrypick=True, is_child_of=False)
                other.neighbor_connections.add(neighbor)
                return
            else:
                neighbor = Neighbor(neighbor=other, sim=is_similar, bit_sim=bit_sim_level, patch_sim=patch_sim_level, explicit_cherrypick=self.other_is_in_my_cherries(other), is_child_of=False)

        self.neighbor_connections.add(neighbor)

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
        if not self.has_explicit_cherries:
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

    # break down patch_set into signatures of hunks
    def signature_patch_set(self, patch_set: unidiff.PatchSet) -> set[int]:
        ret = set()
        for patched_file in patch_set:
            file_name: str = patched_file.source_file + patched_file.target_file
            if patched_file.is_binary_file:
                # use the binary file hash for its own signature else fallback to commit.message
                ret.add(sign_hunk(file_name + "\n" + next((line for line in patched_file.patch_info if line.startswith("index ")), self.commit_message)))
            else:
                ret = ret.union({sign_hunk(file_name + "\n" + get_hunk_string(hunk)) for hunk in patched_file})
        return ret


# Neighbor: a class for the edges of our Neighbor-Graph
# used for the different edge types and edge weights
@dataclass
class Neighbor:
    neighbor: Commit
    sim: bool
    bit_sim: float
    patch_sim: float
    explicit_cherrypick: bool
    is_child_of: bool

    def __hash__(self) -> int:
        return hash(self.neighbor.commit_id)

    def __eq__(self, item) -> bool:
        return item == self.neighbor
