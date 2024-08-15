class CherryReap:
    def __init__(self, reaper, cherries, missing_cherries):
        self.reaper = reaper
        self.cherries = cherries
        self.missing_cherry_ids = missing_cherries
        self.is_complete = len(self.missing_cherry_ids) == 0

    def __eq__(self, other):
        return self.vital_info() == other.vital_info()

    def __hash__(self):
        return hash(self.vital_info())

    def vital_info(self):
        return str(self.reaper.commit_id) + "".join([str(c.commit_id) for c in self.cherries]) + "".join([str(c) for c in self.missing_cherry_ids])
