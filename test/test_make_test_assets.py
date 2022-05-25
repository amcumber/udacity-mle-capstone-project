from pathlib import Path

from typing import List, Tuple
import unittest

import test.make_test_assets


class TestMain(unittest.TestCase):
    """Tests Main Entrypoint"""

    def set_up(self) -> Tuple[Path, List[Path]]:
        self.MAX_SIZE = 10
        root = Path(__file__).parent
        sub_folder_name = "assets"
        file_names = [
            "test_portfolio.json",
            "test_profile.json",
            "test_transcript.json",
        ]
        sub_folder = root.joinpath(sub_folder_name)
        files = [root.joinpath(sub_folder_name, file) for file in file_names]
        self.test_files = files
        self.asset_folder = sub_folder
        return sub_folder, files

    def tearDown(self):
        for file in self.test_files:
            file.unlink(missing_ok=True)
        if list(self.asset_folder.glob("*")):
            raise FileExistsError(f"Files Exist in {self.asset_folder}")
        self.asset_folder.rmdir()

    def verify_file(self, file):
        assert file.exists(), f"missing {file}"

        with file.open("r", encoding="utf-8") as fh:
            n_lines = len(fh.readlines())
            msg = f"file not truncated for file {file}, n={n_lines}"
            assert n_lines <= self.MAX_SIZE, msg

    def test_main(self):
        test.make_test_assets.main()
        sub_folder, files = self.set_up()

        assert sub_folder.exists(), f"{sub_folder} not created!"

        for file in files:
            self.verify_file(file)
