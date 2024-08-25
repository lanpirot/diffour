# tests/test_find_cherries.py
import os.path
import shutil
from unittest import TestCase
import time
from src import find_cherries


def redo_command_string(add_complete_parent_relation: bool) -> None:
    find_cherries.add_complete_parent_relation = add_complete_parent_relation
    find_cherries.no_merges1 = "" if add_complete_parent_relation else " --no-merges"
    find_cherries.no_merges2 = " -m" if add_complete_parent_relation else ""
    find_cherries.git_command = f"git log --all{find_cherries.no_merges1} --date-order --pretty=format:\"{find_cherries.pretty_format}\" -p{find_cherries.no_merges2} -U3 -n {find_cherries.commit_limit}"


class Test(TestCase):
    def setUp(self):
        self.repo_folder = "../data/cherry_repos/"
        if not os.path.isdir(self.repo_folder):
            self.repo_folder = "../" + self.repo_folder
        pass

    def test_analyze_repo(self):
        redo_command_string(add_complete_parent_relation=False)
        start_time = time.time()
        commits = find_cherries.analyze_repo(self.repo_folder + "pydriller", 1000)
        end_time = time.time()
        self.assertTrue(end_time - start_time < 5)
        self.assertEqual(745, len(commits))
        self.assertEqual(6, sum([len([n for n in c.neighbor_connections if n.is_child_of]) for c in commits]))
        self.assertEqual(7, sum([len([n for n in c.neighbor_connections if not n.is_child_of and not n.explicit_cherrypick]) for c in commits]))
        self.assertEqual(sum([len(c.explicit_cherries) for c in commits]),
                         sum([len([n for n in c.neighbor_connections if n.explicit_cherrypick]) for c in commits]))

        redo_command_string(add_complete_parent_relation=True)
        start_time = time.time()
        commits = find_cherries.analyze_repo(self.repo_folder + "pydriller", 1000)
        end_time = time.time()
        self.assertTrue(end_time - start_time < 5)
        self.assertEqual(874, len(commits))
        self.assertEqual(1002, sum([len([n for n in c.neighbor_connections if n.is_child_of]) for c in commits]))
        self.assertEqual(10, sum([len([n for n in c.neighbor_connections if not n.is_child_of and not n.explicit_cherrypick]) for c in commits]))
        self.assertEqual(sum([len(c.explicit_cherries) for c in commits]),
                         sum([len([n for n in c.neighbor_connections if n.explicit_cherrypick]) for c in commits]))

        redo_command_string(add_complete_parent_relation=False)
        start_time = time.time()
        commits = find_cherries.analyze_repo(self.repo_folder + "FFmpeg", 1000)
        end_time = time.time()
        self.assertTrue(end_time - start_time < 5)
        self.assertEqual(1000, len(commits))
        self.assertEqual(600, sum([len([n for n in c.neighbor_connections if n.is_child_of]) for c in commits]))
        self.assertEqual(896, sum([len([n for n in c.neighbor_connections if not n.is_child_of and not n.explicit_cherrypick]) for c in commits]))
        self.assertEqual(sum([len(c.explicit_cherries) for c in commits]),
                         sum([len([n for n in c.neighbor_connections if n.explicit_cherrypick]) for c in commits]))

        redo_command_string(add_complete_parent_relation=True)
        start_time = time.time()
        commits = find_cherries.analyze_repo(self.repo_folder + "FFmpeg", 1000)
        end_time = time.time()
        self.assertTrue(end_time - start_time < 5)
        self.assertEqual(1000, len(commits))
        self.assertEqual(991, sum([len([n for n in c.neighbor_connections if n.is_child_of]) for c in commits]))
        self.assertEqual(896, sum([len([n for n in c.neighbor_connections if not n.is_child_of and not n.explicit_cherrypick]) for c in commits]))
        self.assertEqual(sum([len(c.explicit_cherries) for c in commits]),
                         sum([len([n for n in c.neighbor_connections if n.explicit_cherrypick]) for c in commits]))

    def test_main(self):
        find_cherries.commit_limit = 10
        redo_command_string(add_complete_parent_relation=False)
        find_cherries.repo_folder = self.repo_folder
        find_cherries.full_sample = False
        start_time = time.time()
        find_cherries.main()
        end_time = time.time()
        self.assertTrue(1 < end_time - start_time < 10)

        find_cherries.commit_limit = 1000
        find_cherries.full_sample = True
        redo_command_string(add_complete_parent_relation=False)
        start_time = time.time()
        find_cherries.main()
        end_time = time.time()
        self.assertTrue(10 < end_time - start_time < 60)

    def test_init_git(self):
        tmp_dir = "my_temp_dir"
        os.makedirs(tmp_dir, exist_ok=True)
        os.chdir(tmp_dir)
        find_cherries.init_git()
        gitattributes_file = ".gitattributes"
        self.assertTrue(os.path.isfile(gitattributes_file))
        with open(gitattributes_file) as gf:
            content = gf.read()
            self.assertTrue("* text=auto" in content)
            self.assertTrue("*.pdf binary" in content)
        os.chdir("..")
        shutil.rmtree(tmp_dir)

    def test_parse_git_output(self):
        pass

    def test_get_branch_dict(self):
        pass

    def test_read_in_commits_from_stdout(self):
        pass

    def test_get_any(self):
        pass

    def test_bucketize_buckets(self):
        pass

    def test_remove_singles(self):
        pass

    def test_get_candidate_groups(self):
        pass

    def test_create_commit_id_to_commit(self):
        pass

    def test_create_alt_id_to_commit(self):
        pass

    def test_test_connect_all(self):
        pass

    def test_connect_similar_neighbors(self):
        pass

    def test_connect_cherry_picks(self):
        pass

    def test_connect_parents(self):
        pass

    def test_remove_single_commits(self):
        pass

    def test_how_many_connections_are_known(self):
        pass

    def test_commits_to_csv(self):
        pass

    def test_save_graph(self):
        pass

    def test_remove_duplicate_commits(self):
        pass
