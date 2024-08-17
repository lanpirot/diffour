import os
import subprocess

import cherry_reap
import commit
import time
from joblib import Parallel, delayed

commit_limit = 1000
repo_folder = "../data/cherry_repos/"
save_folder = "cherry_data/"
diff_file = 'diffs_' + str(commit_limit)


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
    # save the original bitmask
    bit_masks = [(bm, bm) for bm in bit_masks]
    for i in range(commit.bit_mask_length):
        bit_masks = [(commit.rotate_left(bm[0]), bm[1]) for bm in bit_masks]
        bit_masks = sorted(bit_masks, key=lambda x: x[0])
        for j in range(len(bit_masks)):
            if commit.is_similar_bitmask(bit_masks[j][0], bit_masks[(j + 1) % len(bit_masks)][0]):
                mi, ma = min(bit_masks[j][1], bit_masks[j + 1][1]), max(bit_masks[j][1], bit_masks[j + 1][1])
                # we found commits with neighboring bitmasks after some rotation, they also have highly similar bitmasks
                # add the groups of each representative to the other representative's group
                all_groups[mi] = all_groups[mi].union(identical_commit_hashes[ma])
                all_groups[ma] = all_groups[ma].union(identical_commit_hashes[mi])

    # remove commits without partners
    candidate_groups = {k: all_groups[k] for k in all_groups if len(all_groups[k]) > 1}
    return candidate_groups


def find_explicit_cherryreaps(candidate_groups):
    explicit_cherryreaps = set()
    for cg in candidate_groups.values():
        # find all commits with explicit cherrypicks
        exp_cps = [c for c in cg if c.has_explicit_cherrypick()]
        for cp in exp_cps:
            (cherries, missing_cherries) = cp.get_all_cherries_in_group(cg)
            reap = cherry_reap.CherryReap(cp, cherries, missing_cherries)
            explicit_cherryreaps.add(reap)
    return explicit_cherryreaps


def commit_id_to_commitf(commits):
    return {c.commit_id: c for c in commits}


def add_close_levenshteins_to_graph(candidate_groups, c_id_to_c):
    for cg in candidate_groups.values():
        lcg = list(cg)
        for i in range(len(lcg) - 1):
            for j in range(i + 1, len(lcg)):
                mi, ma = lcg[i].get_ordered_commit_pair(lcg[j])
                if mi.has_similar_text_to(ma):
                    mi.add_neighbor(ma)
                    ma.add_neighbor(mi)


def add_known_cherry_picks_to_graph(commit_diffs, c_id_to_c):
    for cd in commit_diffs:
        if cd.has_explicit_cherrypick():
            for cherry_id in cd.get_explicit_cherrypicks():
                if cherry_id in c_id_to_c:
                    cherry = c_id_to_c[cherry_id]
                else:
                    cherry = commit.dummy_cherry_commit(cherry_id, diff_marker)
                cd.add_neighbor(cherry)
                cherry.add_neighbor(cd)


def remove_single_commits(commit_diffs):
    cds = []
    for cd in commit_diffs:
        if cd.neighbor_connections:
            cds.append(cd)
    return cds


def how_many_connections_are_known(commit_diffs, folder):
    known, unknown = 0, 0
    for cd in commit_diffs:
        for nc in cd.neighbor_connections:
            if nc.explicit_cherrypick:
                known += 1
            else:
                unknown += 1
    known, unknown = known // 2, unknown // 2
    print(f"{folder}: Known connections (cherry and reaper, or reapers pointing to same cherries): {known}, unknown: {unknown}")


# only save connections from younger commit to older commit (direction of picking), their similarities, whether they have a known connection
def commits_to_csv(commits):
    csv = ""
    for c in commits:
        for cn in c.neighbor_connections:
            if c.is_younger_than(cn.neighbor):
                csv += f"{c.commit_id},{cn.neighbor.commit_id},{cn.bit_sim},{cn.levenshtein_sim},{cn.explicit_cherrypick}\n"
    return csv


def save_cherries(commits, project_name):
    os.makedirs(save_folder, exist_ok=True)
    with open(save_folder + project_name + "_" + str(commit_limit) + ".csv", 'w') as file:
        file.write("reaper,cherry,bit_similar,levenshtein_similar,known_pick\n")
        file.write(commits_to_csv(commits))


def analyze_repo(folder):
    sh_folder = folder.split("/")[-1]
    print(f"Working on {sh_folder} ...")
    #TODO: file_rename_scheme = get_rename_scheme(folder)
    commit_diff_string = create_git_diffs(folder)
    commits = parse_commit_diff_string(commit_diff_string)
    commit_id_to_commit = commit_id_to_commitf(commits)

    #remove non-parseable commits
    parseable_commits = [cd for cd in commits if cd.parseable]
    print(
        f"{sh_folder}: #parseable {len(parseable_commits)} of {len(commits)} commits, #reapers identifiable by commit-message: {sum([1 for cd in parseable_commits if cd.has_explicit_cherrypick()])}")

    candidate_groups = get_candidate_groups(parseable_commits)

    add_close_levenshteins_to_graph(candidate_groups, commit_id_to_commit)
    add_known_cherry_picks_to_graph(commits, commit_id_to_commit)
    final_commits = remove_single_commits(commits)
    how_many_connections_are_known(final_commits, sh_folder)
    #TODO: combine similarity, add bit_similarity, levenshtein_similarity
    #TODO: add parent relation

    #TODO: for those without known connection: look within commit messages for words of length 40 (see githash), print those out
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


if __name__ == '__main__':
    subfolders = os.walk(repo_folder).__next__()[1]
    subfolders = [repo_folder + folder for folder in subfolders]

    small_sample = False

    if small_sample:
        subfolder = repo_folder + "intellij-community"
        #subfolder = repo_folder + "pydriller"
        analyze_repo(subfolder)
    else:
        start_time = time.time()
        Parallel(n_jobs=-1)(delayed(analyze_repo)(repo) for repo in subfolders)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.1f} seconds")
