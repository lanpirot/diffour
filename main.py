import os
import subprocess
import pandas as pd
import commit
import time
from joblib import Parallel, delayed


limit = 1000
repo_folder = "../data/cherry_repos/"
all_diffs = 'diffs_' + str(limit)

commit_marker = "====####====####"
diff_marker = "--- --- --- ---"
pretty_format = commit_marker + "%n%P%n%H%n%an%n%s%b%n" + diff_marker
command = f'git log --all --no-merges --pretty=format:"{pretty_format}" -p -U3 -n' + str(limit)


#create a (long) string of all commits and their unified diffs
def create_git_diffs(folder):
    old_folder = os.getcwd()
    os.chdir(folder)

    if all_diffs in os.listdir():
        #we found the diff file, just read it
        with open(all_diffs, "r", encoding="utf8", errors="replace") as file:
            os.chdir(old_folder)
            return file.read()

    #no diff file found, produce it (and save it)
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace')
    with open(all_diffs, "w+", encoding='utf-8') as file:
        file.write(result.stdout)
    os.chdir(old_folder)
    return result.stdout


def parse_commit_diff_string(commit_diff_string):
    diffs = []
    for cm in commit_diff_string.split(commit_marker + "\n")[1:]:
        diffs += [commit.Commit(cm, diff_marker)]
    return diffs


def find_cherries(commit_diffs):
    pass

    #commit_diffs[c].patch_set[f][h][l].is_context


def analyze_repo(folder):
    print(f"Working on {folder}:")
    # rename_scheme = get_rename_scheme(folder)
    commit_diff_string = create_git_diffs(folder)


    commit_diffs = parse_commit_diff_string(commit_diff_string)
    print(folder, sum([1 for cd in commit_diffs if cd.parseable]), len(commit_diffs), len(commit_diffs) - len(set([cd.get_bit_mask() for cd in commit_diffs])), sum([1 for cd in commit_diffs if cd.claims_cherry_pick()]))
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
        # subfolder = repo_folder + "odoo"
        subfolder = repo_folder + "pydriller"
        analyze_repo(subfolder)
    else:
        start_time = time.time()
        Parallel(n_jobs=-1)(delayed(analyze_repo)(repo) for repo in subfolders)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time} seconds")
