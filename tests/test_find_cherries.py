# tests/test_find_cherries.py
import os.path
from unittest import TestCase, mock
import time
import importlib
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

        find_cherries.full_sample = True
        redo_command_string(add_complete_parent_relation=False)
        start_time = time.time()
        find_cherries.main()
        end_time = time.time()
        self.assertTrue(1 < end_time - start_time < 30)
