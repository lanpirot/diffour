import unidiff
import re

#is_binary_file
#source_file
#target_file


cherry_commit_message_pattern = r"\(cherry picked from commit [a-fA-F0-9]{40}\)"


class Commit:
    def __init__(self, commit_str, diff_marker):
        self.parent_id = None
        self.is_root = None
        self.commit_id = None
        self.author = None
        self.commit_message = None
        self.parseable = True
        self.patch_set = None
        self.parse_commit_str(commit_str, diff_marker)

    def parse_commit_str(self, commit_str, diff_marker):
        commit_str = commit_str.split(diff_marker + "\n")
        if len(commit_str) != 2:
            if commit_str[0].splitlines()[0]:
                raise
            #create dummy commit string for root
            commit_str.append('')
        start = commit_str[0].split("\n", 3)


        if len(start) < 4:
            raise
        (parent_id, commit_id, author, commit_message) = (start[0], start[1], start[2], start[3:][0])

        commit_diff = commit_str[1]
        if len(commit_diff) > 1 and commit_diff[-2] == commit_diff[-1] == "\n":
            commit_diff = commit_diff[:-1]
        try:
            patch_set = unidiff.PatchSet(commit_diff)
        except unidiff.UnidiffParseError as e:
            self.parseable = False
            patch_set = None
        self.init(parent_id, commit_id, author, commit_message, patch_set)

    def init(self, parent_id, commit_id, author, commit_message, patch_set):
        self.parent_id = parent_id
        self.is_root = len(self.parent_id) == 0
        self.commit_id = commit_id
        self.author = author
        self.commit_message = commit_message
        self.patch_set = patch_set

    def claims_cherry_pick(self):
        return re.search(cherry_commit_message_pattern, self.commit_message)

    def get_diff_string(self):
        return self.commit_message + self.patch_set
