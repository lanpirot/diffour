import os
import subprocess
import numpy as np
import re
import commit
import time
from joblib import Parallel, delayed

# A program to seek cherrypicks in git repositories.
# We find explicit cherrypicks (easy), and implicit cherrypicks (very hard).
# Order of action:
#       1. use `git log` to output commits and their udiff
#       2. bite off 10MB chunks of this log for further processing
#       3. in the log, identify commits
#       4. for each commit, parse its basic information (id, author, message) and udiff
#       5. compute a weighted SimHash of the udiff (put commits into buckets based on their SimHash signature)
#       6. rotate, resort buckets, find close neighbors and unionize those buckets
#       7. within each bucket: compare Levenshtein-similarity of all members
#       8. compute graph with edge types:
#           - explicit_cherry_pick (directed)
#           - git_child_and_parent (directed)
#           - highly_similar_commits (undirected, weight1: bit_similarity, weight2: Levenshtein_similarity)

full_sample: bool = False                    # a complete run, or only a test run with a small sample size?
add_complete_parent_relation: bool = False  # store complete git graph (by parent relation), or only parent-relation for relevant nodes?
commit_limit: int = 100000                  # max number of commits of a repository, we sample

repo_folder: str = "../data/cherry_repos/"
save_folder: str = "cherry_data/"
diff_file: str = "diffs_" + str(commit_limit)

commit_marker: str = "====xxx_next_commit_xxx===="
diff_marker: str = "####xxx_next_diff_xxx####"
pretty_format: str = commit_marker + "%n%P%n%H%n%an%n%s%b%n" + diff_marker

no_merges: str = "" if add_complete_parent_relation else " --no-merges"
git_command: str = f"git log --all{no_merges} --date-order --pretty=format:\"{pretty_format}\" -p -U3 -n {commit_limit}"


# prepare each git repo, and .gitattributes file
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


# create a list of commits from the git log output
def parse_git_output(folder: str) -> list[commit.Commit]:
    old_folder: str = os.getcwd()
    os.chdir(folder)
    init_git()
    process: subprocess.Popen = subprocess.Popen(git_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                                 text=True, encoding='utf-8', errors='replace')
    commits: list[commit.Commit] = read_in_commits_from_stdout(process)
    os.chdir(old_folder)
    return commits


# git log output is sometimes gigantic, bite off chunks to parse
# this saves space (don't save complete output in memory)
# this saves time (don't bite off line by line)
def read_in_commits_from_stdout(process: subprocess.Popen) -> list[commit.Commit]:
    commits: list[commit.Commit] = []
    remnant: str = ""
    commit_marker_newline: str = commit_marker + "\n"
    while True:
        chunk: str = process.stdout.read(2 ** 20 * 10)  # Read up to 10MB at a time
        if not chunk:
            commits.append(commit.Commit(remnant, diff_marker))
            break
        if commit_marker_newline not in chunk:
            continue

        # Prepend any leftover data from the previous chunk
        chunk = remnant + chunk
        new_commits = chunk.split(commit_marker_newline)
        for nc in new_commits[:-1]:
            if nc:
                commits.append(commit.Commit(nc, diff_marker))
        remnant = new_commits[-1]
    process.stdout.close()
    process.wait()
    return commits


# rotate bits of each bucket, find closely related buckets and unionize them
# only unionize the original buckets on top, though -- this avoids giant buckets of actually unrelated content
def get_candidate_groups(commit_diffs: list[commit.Commit]) -> dict[int, set[commit.Commit]]:
    identical_commit_hashes: dict[int, set[commit.Commit]] = {cd.bit_mask: set() for cd in commit_diffs}
    for cd in commit_diffs:
        identical_commit_hashes[cd.bit_mask].add(cd)

    all_groups: dict[int, set[commit.Commit]] = identical_commit_hashes.copy()

    bit_masks_keys: list[int] = list(identical_commit_hashes.keys())
    # save the original bitmask
    bit_masks: list[tuple[int, int]] = [(bm, bm) for bm in bit_masks_keys]
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


# create a lookup table from commit_id to commits
def create_commit_id_to_commit(commits: list[commit.Commit]) -> dict[str, commit.Commit]:
    return {c.commit_id: c for c in commits}


# add a connection to our graph for all commits, that we deem highly similar, based on their udiff
def connect_similar_neighbors(candidate_groups: dict[int, set[commit.Commit]]) -> None:
    for cg in candidate_groups.values():
        lcg: list[commit.Commit] = list(cg)
        for i in range(len(lcg) - 1):
            for j in range(i + 1, len(lcg)):
                mi, ma = lcg[i].get_ordered_commit_pair(lcg[j])
                mi.add_neighbor(ma)
                ma.add_neighbor(mi)


# add a connection to our graph for each picker and its cherries
def connect_cherry_picks(commit_diffs: list[commit.Commit], c_id_to_c: dict[str, commit.Commit]) -> None:
    for cd in commit_diffs:
        if cd.has_explicit_cherrypick():
            for cherry_id in cd.explicit_cherries:
                cherry: commit.Commit
                if cherry_id in c_id_to_c:
                    cherry = c_id_to_c[cherry_id]
                else:
                    cherry = commit.dummy_cherry_commit(cherry_id, diff_marker)
                    # it would be cleaner to add dummies to the c_id_to_c lookup table
                cd.add_neighbor(cherry)
                cherry.add_neighbor(cd)


# add a connection to our graph for each parent and child, based on git commit ids and their parent-relation
def connect_parents(commits: list[commit.Commit], c_id_to_c: dict[str, commit.Commit]) -> None:
    for c in commits:
        if c.parent_id in c_id_to_c:
            c.add_neighbor(c_id_to_c[c.parent_id])


# remove all commits without a neighbor. they are a bit boring for our purposes
def remove_single_commits(commit_diffs: list[commit.Commit]) -> list[commit.Commit]:
    return [cd for cd in commit_diffs if cd.neighbor_connections]


# give a report about our graph and its edges
def how_many_connections_are_known(commit_diffs: list[commit.Commit], folder: str) -> None:
    known: int = 0
    unknown: int = 0
    for cd in commit_diffs:
        for nc in cd.neighbor_connections:
            if nc.explicit_cherrypick:
                known += 1
            else:
                unknown += 1
    known, unknown = known // 2, unknown // 2
    print(f"{folder}: Known connections (cherry and picker): {known}, unknown: {unknown}")


# only save connections from younger commit (picker, child) to older commit (cherry, parent), their similarities, whether they have a known connection
def commits_to_csv(commits: list[commit.Commit]) -> str:
    csv: str = ""
    for c in commits:
        for cn in c.neighbor_connections:
            if c.is_younger_than(cn.neighbor):
                csv += f"{c.commit_id},{cn.neighbor.commit_id},{cn.sim},{cn.bit_sim},{cn.levenshtein_sim},{cn.explicit_cherrypick},{np.nan},{cn.is_child_of}\n"
    return csv


# save the graph to disk
def save_graph(commits: list[commit.Commit], project_name: str) -> None:
    os.makedirs(save_folder, exist_ok=True)
    with open(save_folder + project_name + "_" + str(commit_limit) + ".csv", 'w') as file:
        file.write("tail(picker;child),head(cherry;parent),similar,bit_similarity,levenshtein_similarity,picks_explicitly,patch_similarity,is_child_of\n")
        file.write(commits_to_csv(commits))


# main loop, works in order as described above
def analyze_repo(folder: str) -> None:
    job_start_time: float = time.time()
    sh_folder: str = folder.split("/")[-1]
    print(f"Working on {sh_folder} ...")
    # TODO: file_rename_scheme = get_rename_scheme(folder)
    commits: list[commit.Commit] = parse_git_output(folder)
    commit_id_to_commit: dict[str, commit.Commit] = create_commit_id_to_commit(commits)

    # remove non-parseable commits
    parseable_commits: list[commit.Commit] = [cd for cd in commits if cd.parseable]
    print(f"{sh_folder}: #parseable {len(parseable_commits)} of {len(commits)} commits, "
          f"#explicit pickers: {sum([1 for cd in parseable_commits if cd.has_explicit_cherrypick()])}")

    candidate_groups: dict[int, set[commit.Commit]] = get_candidate_groups(parseable_commits)

    connect_similar_neighbors(candidate_groups)
    connect_cherry_picks(commits, commit_id_to_commit)

    if add_complete_parent_relation:
        connect_parents(commits, commit_id_to_commit)
        final_commits: list[commit.Commit] = remove_single_commits(commits)
    else:
        final_commits: list[commit.Commit] = remove_single_commits(commits)
        connect_parents(final_commits, commit_id_to_commit)

    how_many_connections_are_known(final_commits, sh_folder)
    # TODO: why are some known cherry pickers not in the final_commits or have wrong edge type?

    # TODO: for those without known connection: look within commit messages for words of length 40 (see git hash), print those out
    # goal: find all other reference systems, people use
    # import re
    # git_hash40 = r"[a-fA-F0-9]{40}"
    # for c in final_commits:
    #     for n in c.neighbor_connections:
    #         if n.claimed_cherry_connection:
    #             continue
    #         nn = n.neighbor
    #         if re.search(git_hash40, c.commit_message) or re.search(git_hash40, nn.commit_message):
    #             print(c.commit_id, "\n", c.commit_message, "\n\n\n", nn.commit_id, "\n", nn.commit_message, "\n\n\n")
    save_graph(final_commits, sh_folder)
    pass
    job_end_time: float = time.time()
    print(f"{sh_folder}: Execution time: {job_end_time - job_start_time:.1f} seconds")


# outer call starting a subprocess for each repository
if __name__ == '__main__':
    subfolders: list[str] = os.walk(repo_folder).__next__()[1]
    subfolders: list[str] = [repo_folder + folder for folder in subfolders]

    if full_sample:
        start_time: float = time.time()
        Parallel(n_jobs=-1)(delayed(analyze_repo)(repo) for repo in subfolders)
        end_time: float = time.time()
        print(f"Execution time: {end_time - start_time:.1f} seconds")
    else:
        subfolder: str = repo_folder + "odoo"
        analyze_repo(subfolder)
