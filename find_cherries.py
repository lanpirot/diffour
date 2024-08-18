import os
import subprocess
import numpy as np
import re

import commit
import time
from joblib import Parallel, delayed

commit_limit: int = 1000
repo_folder: str = "../data/cherry_repos/"
save_folder: str = "cherry_data/"
diff_file: str = "diffs_" + str(commit_limit)

commit_marker: str = "====xxx_next_commit_xxx===="
commit_markern: str = commit_marker + "\n"
diff_marker: str = "####xxx_next_diff_xxx####"
pretty_format: str = commit_marker + "%n%P%n%H%n%an%n%s%b%n" + diff_marker
command: str = f"git log --all --no-merges --date-order --pretty=format:\"{pretty_format}\" -p -U3 -n {commit_limit}"


def init_git() -> None:
    subprocess.run("git stash clear", shell=True)
    with open(".gitattributes", "r+") as file:
        text: str = file.read()
        if not re.search(r"\*\.pdf\s*binary", text):
            text += "*.pdf binary\n"
        if not re.search(r"\*\s*text\s*=\s*auto", text):
            text += "* text=auto\n"
    with open(".gitattributes", "w") as file:
        file.write(text)


# create a (long) string of all commits and their unified diffs
def parse_git_output(folder: str) -> list[commit.Commit]:
    old_folder: str = os.getcwd()
    os.chdir(folder)
    init_git()
    process: subprocess.Popen = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                                 text=True, encoding='utf-8', errors='replace')
    commits: list[commit.Commit] = read_in_commits_from_stdout(process)
    os.chdir(old_folder)
    return commits


def read_in_commits_from_stdout(process: subprocess.Popen) -> list[commit.Commit]:
    commits: list[commit.Commit] = []
    remnant: str = ""
    while True:
        chunk: str = process.stdout.read(2**20 * 10)  # Read up to 10MB at a time
        if not chunk:
            commits.append(commit.Commit(remnant, diff_marker))
            break
        if commit_markern not in chunk:
            continue

        # Prepend any leftover data from the previous chunk
        chunk = remnant + chunk
        new_commits = chunk.split(commit_markern)
        for nc in new_commits[:-1]:
            if nc:
                commits.append(commit.Commit(nc, diff_marker))
        remnant = new_commits[-1]
    process.stdout.close()
    process.wait()
    return commits


def get_candidate_groups(commit_diffs: list[commit.Commit]) -> dict[int, set[commit.Commit]]:
    identical_commit_hashes: dict[int, set[commit.Commit]] = {cd.bit_mask: set() for cd in commit_diffs}
    for cd in commit_diffs:
        identical_commit_hashes[cd.bit_mask].add(cd)

    all_groups: dict[int, set[commit.Commit]] = identical_commit_hashes.copy()

    bit_masks_keys: list[int] = list(identical_commit_hashes.keys())
    # save the original bitmask
    bit_masks: list[(int, int)] = [(bm, bm) for bm in bit_masks_keys]
    for i in range(commit.bit_mask_length):
        bit_masks = [(commit.rotate_left(bm[0]), bm[1]) for bm in bit_masks]
        bit_masks = sorted(bit_masks, key=lambda x: x[0])
        for j in range(len(bit_masks)):
            if commit.is_similar_bitmask(bit_masks[j][0], bit_masks[(j + 1) % len(bit_masks)][0])[0]:
                mi, ma = min(bit_masks[j][1], bit_masks[j + 1][1]), max(bit_masks[j][1], bit_masks[j + 1][1])
                # we found commits with neighboring bitmasks after some rotation, they also have highly similar bitmasks
                # add the groups of each representative to the other representative's group
                all_groups[mi] = all_groups[mi].union(identical_commit_hashes[ma])
                all_groups[ma] = all_groups[ma].union(identical_commit_hashes[mi])

    # remove commits without partners
    candidate_groups: dict[int, set[commit.Commit]] = {k: all_groups[k] for k in all_groups if len(all_groups[k]) > 1}
    return candidate_groups


def create_commit_id_to_commit(commits):
    return {c.commit_id: c for c in commits}


def connect_similar_neighbors(candidate_groups):
    for cg in candidate_groups.values():
        lcg = list(cg)
        for i in range(len(lcg) - 1):
            for j in range(i + 1, len(lcg)):
                mi, ma = lcg[i].get_ordered_commit_pair(lcg[j])
                mi.add_neighbor(ma)
                ma.add_neighbor(mi)


def connect_cherry_picks(commit_diffs, c_id_to_c):
    for cd in commit_diffs:
        if cd.has_explicit_cherrypick():
            for cherry_id in cd.explicit_cherries:
                if cherry_id in c_id_to_c:
                    cherry = c_id_to_c[cherry_id]
                else:
                    cherry = commit.dummy_cherry_commit(cherry_id, diff_marker)
                    # it would be cleaner to add dummies to the c_id_to_c lookup table
                cd.add_neighbor(cherry)
                cherry.add_neighbor(cd)


def remove_single_commits(commit_diffs):
    return [cd for cd in commit_diffs if cd.neighbor_connections]


def how_many_connections_are_known(commit_diffs, folder):
    known, unknown = 0, 0
    for cd in commit_diffs:
        for nc in cd.neighbor_connections:
            if nc.explicit_cherrypick:
                known += 1
            else:
                unknown += 1
    known, unknown = known // 2, unknown // 2
    print(f"{folder}: Known connections (cherry and picker): {known}, unknown: {unknown}")


# only save connections from younger commit to older commit (direction of picking), their similarities, whether they have a known connection
def commits_to_csv(commits):
    csv = ""
    for c in commits:
        for cn in c.neighbor_connections:
            if c.is_younger_than(cn.neighbor):
                csv += f"{c.commit_id},{cn.neighbor.commit_id},{cn.sim},{cn.bit_sim},{cn.levenshtein_sim},{cn.explicit_cherrypick},{np.nan}\n"
    return csv


def save_cherries(commits, project_name):
    os.makedirs(save_folder, exist_ok=True)
    with open(save_folder + project_name + "_" + str(commit_limit) + ".csv", 'w') as file:
        file.write("picker,cherry,similar,bit_similarity,levenshtein_similarity,known_pick,patch_similarity\n")
        file.write(commits_to_csv(commits))


def analyze_repo(folder):
    job_start_time = time.time()
    sh_folder = folder.split("/")[-1]
    print(f"Working on {sh_folder} ...")
    # TODO: file_rename_scheme = get_rename_scheme(folder)
    commits = parse_git_output(folder)
    commit_id_to_commit = create_commit_id_to_commit(commits)

    # remove non-parseable commits
    parseable_commits = [cd for cd in commits if cd.parseable]
    print(f"{sh_folder}: #parseable {len(parseable_commits)} of {len(commits)} commits, "
          f"#explicit pickers: {sum([1 for cd in parseable_commits if cd.has_explicit_cherrypick()])}")

    candidate_groups = get_candidate_groups(parseable_commits)

    connect_similar_neighbors(candidate_groups)
    connect_cherry_picks(commits, commit_id_to_commit)
    # TODO: add parent relation (add merges back in)
    # connect_parents(commits, commit_id_to_commit)
    final_commits = remove_single_commits(commits)

    how_many_connections_are_known(final_commits, sh_folder)

    # TODO: for those without known connection: look within commit messages for words of length 40 (see git hash), print those out
    # import re
    # git_hash40 = r"[a-fA-F0-9]{40}"
    # for c in final_commits:
    #     for n in c.neighbor_connections:
    #         if n.claimed_cherry_connection:
    #             continue
    #         nn = n.neighbor
    #         if re.search(git_hash40, c.commit_message) or re.search(git_hash40, nn.commit_message):
    #             print(c.commit_id, "\n", c.commit_message, "\n\n\n", nn.commit_id, "\n", nn.commit_message, "\n\n\n")
    save_cherries(final_commits, sh_folder)
    pass
    job_end_time = time.time()
    print(f"{sh_folder}: Execution time: {job_end_time - job_start_time:.1f} seconds")


if __name__ == '__main__':
    subfolders = os.walk(repo_folder).__next__()[1]
    subfolders = [repo_folder + folder for folder in subfolders]

    full_sample = True

    if full_sample:
        start_time = time.time()
        Parallel(n_jobs=-1)(delayed(analyze_repo)(repo) for repo in subfolders)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.1f} seconds")
    else:
        # subfolder = repo_folder + "intellij-community"
        subfolder = repo_folder + "WebKit"
        analyze_repo(subfolder)
