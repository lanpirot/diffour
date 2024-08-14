import os
import subprocess
import pandas as pd
import commit
import time
from joblib import Parallel, delayed

commit_limit = 10000
repo_folder = "../data/cherry_repos/"
diff_file = 'diffs_' + str(commit_limit)
tolerable_bit_diff = 4

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


def find_cherries(commit_diffs):
    identical_commit_hashes = {cd.bit_mask: set() for cd in commit_diffs}
    for cd in commit_diffs:
        identical_commit_hashes[cd.bit_mask].add(cd)

    candidate_cherries = identical_commit_hashes.copy()

    bit_masks = list(identical_commit_hashes.keys())
    #save the original bitmask
    bit_masks = [(bm,bm) for bm in bit_masks]
    for i in range(commit.bit_mask_length):
        bit_masks = [(commit.rotate_left(bm[0]), bm[1]) for bm in bit_masks]
        bit_masks = sorted(bit_masks, key=lambda x: x[0])
        for j in range(len(bit_masks)):
            if commit.count_same_bits(bit_masks[j][0], bit_masks[(j+1) % len(bit_masks)][0]) + tolerable_bit_diff > commit.bit_mask_length:
                mi, ma = min(bit_masks[j][1], bit_masks[j+1][1]), max(bit_masks[j][1], bit_masks[j+1][1])
                #add the bigger one (and its candidate cherries) into the smaller one's candidate cherry group
                candidate_cherries[mi] = candidate_cherries[mi].union(identical_commit_hashes[ma])

    print(f"total candidate cherries: {sum([len(cd) - 1 for cd in candidate_cherries.values()])}")
    pass


def analyze_repo(folder):
    print(f"Working on {folder}:")
    # rename_scheme = get_rename_scheme(folder)
    commit_diff_string = create_git_diffs(folder)

    commit_diffs = parse_commit_diff_string(commit_diff_string)
    print(f"{folder.split("/")[-1]}:  commits parseable: {sum([1 for cd in commit_diffs if cd.parseable])} of: {len(commit_diffs)} identical hash: {len(commit_diffs) - len(set([cd.get_bit_mask() for cd in commit_diffs]))} (commits, claiming cherry-pick: {sum([1 for cd in commit_diffs if cd.claims_cherry_pick()])})")
    cherry_candidates = find_cherries(commit_diffs)
    # cherries = filter_cherries(cherry_candidates)
    # save_cherries(cherries)
    # break


pass

if __name__ == '__main__':
    subfolders = os.walk(repo_folder).__next__()[1]
    subfolders = [repo_folder + folder for folder in subfolders]

    small_mini = False

    if small_mini:
        subfolder = repo_folder + "WebKit"
        #subfolder = repo_folder + "pydriller"
        analyze_repo(subfolder)
    else:
        start_time = time.time()
        Parallel(n_jobs=-1)(delayed(analyze_repo)(repo) for repo in subfolders)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
