import os
import subprocess

import cherry_reap
import commit
import time
from joblib import Parallel, delayed
import textdistance
import difflib

commit_limit = 10000
repo_folder = "../data/cherry_repos/"
diff_file = 'diffs_' + str(commit_limit)
tolerable_bit_diff = 8

commit_marker = "====xxx_next_commit_xxx===="
diff_marker = "####xxx_next_diff_xxx####"
pretty_format = commit_marker + "%n%P%n%H%n%an%n%s%b%n" + diff_marker
command = f'git log --all --no-merges --date-order --pretty=format:"{pretty_format}" -p -U3 -n' + str(commit_limit)


#create a (long) string of all commits and their unified diffs
def create_git_diffs(folder):
    old_folder = os.getcwd()
    os.chdir(folder)

    if diff_file in os.listdir():
        #we found the diff file, just read it
        with open(diff_file, "r", encoding="utf8", errors="replace") as file:
            os.chdir(old_folder)
            return file.read()

    #no diff file found, produce it (and save it)
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                            encoding='utf-8', errors='replace')
    with open(diff_file, "w+", encoding='utf-8') as file:
        file.write(result.stdout)
    os.chdir(old_folder)
    return result.stdout

def parse_commit_diff_string(commit_diff_string):
    diffs = []
    for cm in commit_diff_string.split(commit_marker + "\n")[1:]:
        diffs += [commit.Commit(cm, diff_marker)]
    return diffs


def get_candidate_groups(commit_diffs):
    identical_commit_hashes = {cd.bit_mask: set() for cd in commit_diffs}
    for cd in commit_diffs:
        identical_commit_hashes[cd.bit_mask].add(cd)

    all_groups = identical_commit_hashes.copy()

    bit_masks = list(identical_commit_hashes.keys())
    #save the original bitmask
    bit_masks = [(bm, bm) for bm in bit_masks]
    for i in range(commit.bit_mask_length):
        bit_masks = [(commit.rotate_left(bm[0]), bm[1]) for bm in bit_masks]
        bit_masks = sorted(bit_masks, key=lambda x: x[0])
        for j in range(len(bit_masks)):
            if commit.count_same_bits(bit_masks[j][0], bit_masks[(j + 1) % len(bit_masks)][0]) + tolerable_bit_diff > commit.bit_mask_length:
                mi, ma = min(bit_masks[j][1], bit_masks[j + 1][1]), max(bit_masks[j][1], bit_masks[j + 1][1])
                #we found the neighbors are not only neighbors, but also tolerably close (less than tolerable_bit_diff distance)
                #add the groups of each representative to the other representative's group
                all_groups[mi] = all_groups[mi].union(identical_commit_hashes[ma])
                all_groups[ma] = all_groups[ma].union(identical_commit_hashes[mi])

    #remove commits without partners
    candidate_groups = {k: all_groups[k] for k in all_groups if len(all_groups[k]) > 1}
    print(f"total with potential partners: {sum([len(cd) for cd in candidate_groups.values()])}")
    return all_groups, candidate_groups


def find_claiming_cherry_reaps(candidate_groups):
    claimed_cherry_reaps = set()
    for cg in candidate_groups.values():
        #find all commits that claim to pick cherr(ies)
        claimants = [c for c in cg if c.claims_cherry_pick()]
        for cp in claimants:
            (cherries, missing_cherries) = cp.get_all_cherries_in_group(cg)
            reap = cherry_reap.CherryReap(cp, cherries, missing_cherries)
            claimed_cherry_reaps.add(reap)
    return claimed_cherry_reaps


def analyze_repo(folder):
    print(f"Working on {folder}:")
    # rename_scheme = get_rename_scheme(folder)
    commit_diff_string = create_git_diffs(folder)

    commit_diffs = parse_commit_diff_string(commit_diff_string)
    print(f"{folder.split("/")[-1]}: commits parseable: {sum([1 for cd in commit_diffs if cd.parseable])} of: {len(commit_diffs)}, identical hash: {len(commit_diffs) - len(set([cd.get_bit_mask() for cd in commit_diffs]))}, commits, claiming cherry-pick: {sum([1 for cd in commit_diffs if cd.claims_cherry_pick()])}")
    all_groups, candidate_groups = get_candidate_groups(commit_diffs)
    claimed_cherry_reaps = find_claiming_cherry_reaps(all_groups)

    #unknown_groups = filter_by_similarity(unknown_groups)
    print()
    # save_cherries(cherries)


if __name__ == '__main__':
    subfolders = os.walk(repo_folder).__next__()[1]
    subfolders = [repo_folder + folder for folder in subfolders]

    small_sample = True

    if small_sample:
        subfolder = repo_folder + "intellij-community"
        #subfolder = repo_folder + "pydriller"
        analyze_repo(subfolder)
    else:
        start_time = time.time()
        Parallel(n_jobs=-1)(delayed(analyze_repo)(repo) for repo in subfolders)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
