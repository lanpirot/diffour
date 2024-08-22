import os
import gc
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
#       8. compute graph with edge types: all directed edges are young->old
#           - explicit_cherry_pick (directed)
#           - git_child_and_parent (directed)
#           - highly_similar_commits (directed, weight1: bit_similarity, weight2: Levenshtein_similarity)

full_sample: bool = True  # a complete run, or only a test run with a small sample size?
add_complete_parent_relation: bool = False  # store complete git graph (by parent relation), or only parent-relation for relevant nodes?
commit_limit: int = 10 ** 3  # max number of commits of a repository, we sample
max_bucket_overspill = 1

repo_folder: str = "../data/cherry_repos/"
save_folder: str = "cherry_data/"

commit_marker: str = "====xxx_next_commit_xxx===="
diff_marker: str = "####xxx_next_diff_xxx####"
pretty_format: str = commit_marker + "%n%H%n%P%n%an%n%s%b%n" + diff_marker

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


def get_branch_dict() -> dict[str, str]:
    branch_ids: str = subprocess.run(f"git name-rev --all", capture_output=True, text=True, shell=True).stdout.strip()
    branch_dict: dict[str, str] = dict()
    for commit_branch in branch_ids.splitlines():
        commit_branch = commit_branch.split(" ")
        commit_id, branch = commit_branch[0], commit_branch[1].split("~")[0].split("^")[0]
        branch_dict[commit_id] = branch
    return branch_dict


# git log output is sometimes gigantic, bite off chunks to parse
# this saves space (don't save complete output in memory)
# this saves time (don't bite off line by line)
def read_in_commits_from_stdout(process: subprocess.Popen) -> list[commit.Commit]:
    commits: list[commit.Commit] = []
    remnant: str = ""
    commit.Commit.branch_dict = get_branch_dict()
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
        for new_commit in new_commits[:-1]:
            if new_commit:
                commits.append(commit.Commit(new_commit, diff_marker))
        remnant = new_commits[-1]
    process.stdout.close()
    process.wait()
    return commits


def get_any(s: set):
    return next(iter(s))


# Bucket for built-in Hash
class InnerBuckets:
    def __init__(self, hsh_to_members: dict[int, set[commit.Commit]] = None):
        self.hsh_to_members: dict[int, set[commit.Commit]]
        if hsh_to_members:
            self.hsh_to_members = hsh_to_members
        else:
            self.hsh_to_members = dict()

    def __getitem__(self, index):
        return self.hsh_to_members[index]

    def __len__(self):
        return sum([len(v) for v in self.hsh_to_members.values()])

    def __iter__(self):
        return iter(self.hsh_to_members)

    def values(self):
        return self.hsh_to_members.values()

    def add(self, other_buckets: 'InnerBuckets'):
        self.hsh_to_members = {**self.hsh_to_members, **other_buckets.hsh_to_members}


def bucketize_buckets(commits: list[commit.Commit]) -> dict[int, InnerBuckets]:
    signa_buckets: dict[int, set[commit.Commit]] = {c.commit_lsh_signature: set() for c in commits}
    for c in commits:
        signa_buckets[c.commit_lsh_signature].add(c)
    buckets: dict[int, InnerBuckets] = dict()

    for bm in signa_buckets:
        signa_bucket: set[commit.Commit] = signa_buckets[bm]
        inner_buckets: InnerBuckets = InnerBuckets({c.commit_fine_signature: set() for c in signa_bucket})
        for c in signa_bucket:
            inner_buckets[c.commit_fine_signature].add(c)
        buckets[bm] = inner_buckets
    return buckets


def remove_singles(buckets: dict[int, InnerBuckets]) -> dict[int, InnerBuckets]:
    return {k: v for k, v in buckets.items() if len(v) > 1}


# rotate bits of each bucket, find closely related buckets and unionize them
# only unionize the original buckets on top, though -- this avoids giant buckets of actually unrelated content
def get_candidate_groups(commits: list[commit.Commit]) -> dict[int, InnerBuckets]:
    orig_buckets: dict[int, InnerBuckets] = bucketize_buckets(commits)
    mutable_buckets: dict[int, InnerBuckets] = bucketize_buckets(commits)

    # tuple of rotated and original signatures
    signatures: list[tuple[int, int]] = [(s, s) for s in orig_buckets.keys()]
    for i in range(commit.bit_mask_length):
        signatures = [(commit.rotate_left(bm[0]), bm[1]) for bm in signatures]
        signatures = sorted(signatures, key=lambda x: x[0])
        for j in range(len(signatures)):
            bucket_offset = 1
            while (bucket_offset < max_bucket_overspill + 1 and
                   commit.is_similar_signature(signatures[j][0], signatures[(j + bucket_offset) % len(signatures)][0])[0]):
                mi, ma = (min(signatures[j][1], signatures[(j + bucket_offset) % len(signatures)][1]),
                          max(signatures[j][1], signatures[(j + bucket_offset) % len(signatures)][1]))
                # a neighboring bucket has commits with a highly similar signature, let the buckets overspill:
                mutable_buckets[mi].add(orig_buckets[ma])
                bucket_offset += 1

    # remove commits without partners
    return remove_singles(mutable_buckets)


# create a lookup table from commit_id to commits
def create_commit_id_to_commit(commits: list[commit.Commit]) -> dict[str, commit.Commit]:
    return {c.commit_id: c for c in commits}


# similar lookup table, but not every commit has an alternative ID
def create_alt_id_to_commit(commits: list[commit.Commit]) -> dict[str, commit.Commit]:
    return {c.alt_id: c for c in commits if c.alt_id}


# connect all members of s1 to s2
def connect_all(s1: set[commit.Commit], s2: set[commit.Commit], patch_sim: tuple[bool, float]) -> None:
    for s in s1:
        for t in s2:
            if s == t:
                continue
            mi, ma = s.get_ordered_commit_pair(t)
            ma.add_neighbor(mi, patch_sim)


# add a connection to our graph for all commits, that we deem highly similar, based on their udiff
def connect_similar_neighbors(buckets: dict[int, InnerBuckets]) -> None:
    for ibs in buckets.values():
        for i in ibs.values():
            for j in ibs.values():
                if i == j:
                    connect_all(i, i, (True, 1.0))
                else:
                    patch_sim_b, patch_sim = get_any(i).is_similar_patch_to(get_any(j))
                    if patch_sim_b:
                        connect_all(i, j, (patch_sim_b, patch_sim))


# add a connection to our graph for each picker and its cherries
def connect_cherry_picks(commits: list[commit.Commit], c_id_to_c: dict[str, commit.Commit], alt_id_to_c: dict[str, commit.Commit]) -> None:
    for c in commits:
        if c.has_explicit_cherries:
            for cherry_id in c.explicit_cherries:
                cherry: commit.Commit
                if cherry_id in c_id_to_c:
                    cherry = c_id_to_c[cherry_id]
                elif cherry_id in alt_id_to_c:
                    cherry = alt_id_to_c[cherry_id]
                else:
                    cherry = commit.dummy_cherry_commit(cherry_id, diff_marker)
                    # it would be cleaner to add dummies to the c_id_to_c lookup table
                c.add_neighbor(cherry)


# add a connection to our graph for each parent and child, based on git commit ids and their parent-relation
def connect_parents(commits: list[commit.Commit], c_id_to_c: dict[str, commit.Commit]) -> None:
    for c in commits:
        for p in c.parent_ids:
            if p in c_id_to_c:
                c.add_neighbor(c_id_to_c[p])


# remove all commits without a neighbor. they are a bit boring for our purposes
def remove_single_commits(commits: list[commit.Commit]) -> list[commit.Commit]:
    return [c for c in commits if c.neighbor_connections]


# give a report about our graph and its edges
def how_many_connections_are_known(commits: list[commit.Commit], folder: str) -> None:
    picks: int = 0
    children: int = 0
    unknown: int = 0
    for c in commits:
        picks += sum([1 for n in c.neighbor_connections if n.explicit_cherrypick])
        children += sum([1 for n in c.neighbor_connections if n.is_child_of])
        unknown += sum([1 for n in c.neighbor_connections if not (n.explicit_cherrypick or n.is_child_of)])
    print(f"{folder}: Known connections: picker->cherry {picks}, child->parent {children};  unknown: {unknown}")


# only save connections from younger commit (picker, child) to older commit (cherry, parent), their similarities, whether they have a known connection
def commits_to_csv(commits: list[commit.Commit]) -> str:
    csv: str = ""
    for c in commits:
        for cn in c.neighbor_connections:
            if c.is_younger_than(cn.neighbor):
                csv += f"{c.commit_id},{cn.neighbor.commit_id},{cn.sim},{cn.bit_sim},{cn.patch_sim},{cn.explicit_cherrypick},{np.nan},{cn.is_child_of}\n"
    return csv


def save_graph(commits: list[commit.Commit], project_name: str) -> None:
    os.makedirs(save_folder, exist_ok=True)
    with open(save_folder + project_name + "_" + str(commit_limit) + ".csv", 'w') as file:
        file.write(
            "tail(picker;child),head(cherry;parent),similar,bit_similarity,patch_similarity,picks_explicitly,is_child_of\n")
        file.write(commits_to_csv(commits))


# main loop
def analyze_repo(folder: str) -> None:
    job_start_time: float = time.time()
    sh_folder: str = folder.split("/")[-1]
    print(f"Working on {sh_folder} ...")
    # TODO: file_rename_scheme = get_rename_scheme(folder)
    commits: list[commit.Commit] = parse_git_output(folder)
    commit_id_to_commit: dict[str, commit.Commit] = create_commit_id_to_commit(commits)
    alt_id_to_commit: dict[str, commit.Commit] = create_alt_id_to_commit(commits)

    # remove non-parseable commits
    parseable_commits: list[commit.Commit] = [c for c in commits if c.parseable]
    unparseable_commits: list[commit.Commit] = [c for c in commits if not c.parseable]
    print(f"{sh_folder}: #parseable {len(parseable_commits)} of {len(commits)} commits, "
          f"#explicit pickers: {sum([1 for c in commits if c.has_explicit_cherries])}, "
          f"#explicit picks: {sum([len(c.explicit_cherries) for c in commits if c.has_explicit_cherries])}")

    buckets: dict[int, InnerBuckets] = get_candidate_groups(parseable_commits)
    non_parseable_buckets: dict[int, InnerBuckets] = {(2 ** commit.bit_mask_length + i): InnerBuckets({0: {unparseable_commits[i]}}) for i in
                                                      range(len(unparseable_commits))}
    buckets = {**buckets, **non_parseable_buckets}

    connect_similar_neighbors(buckets)
    connect_cherry_picks(commits, commit_id_to_commit, alt_id_to_commit)
    if add_complete_parent_relation:
        connect_parents(commits, commit_id_to_commit)
    else:
        final_commits: list[commit.Commit] = remove_single_commits(commits)
        f_commit_id_to_commit: dict[str, commit.Commit] = create_commit_id_to_commit(final_commits)
        connect_parents(final_commits, f_commit_id_to_commit)

    # TODO: prune tree: weakest outgoing (non-explicit) edges
    # TODO: prune tree: remove transitive edges pointing to same sink
    how_many_connections_are_known(commits, sh_folder)
    save_graph(commits, sh_folder)
    pass
    job_end_time: float = time.time()
    print(f"{sh_folder}: Execution time: {job_end_time - job_start_time:.1f} seconds")


if __name__ == '__main__':
    gc.collect()
    subfolders: list[str] = os.walk(repo_folder).__next__()[1]
    subfolders: list[str] = [repo_folder + folder for folder in subfolders]

    if full_sample:
        start_time: float = time.time()
        Parallel(n_jobs=-1)(delayed(analyze_repo)(repo) for repo in subfolders)
        end_time: float = time.time()
        print(f"Execution time: {end_time - start_time:.1f} seconds")
    else:
        subfolder: str = repo_folder + "intellij-community"
        analyze_repo(subfolder)
