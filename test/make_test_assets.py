from pathlib import Path

from typing import Any, List, Tuple


def make_asset_folder(directory: Path) -> Path:
    """Make the asset folder in directory location"""
    sub_folder = directory.joinpath("assets")
    sub_folder.mkdir(exist_ok=True)
    return sub_folder


def read_truncated_file(source: Path, max_lines: int = 10) -> List[Any]:
    """Read up to the first lines of a file, specified by max_lines"""
    with open(source, "r", encoding="utf-8") as fh:
        lines = []
        for line in fh:
            lines.append(line)
            if len(lines) >= max_lines:
                return lines
        return lines


def write_file(target: Path, data: Any) -> None:
    """Write contents to ASCII file"""
    with open(target, "w", encoding="utf-8") as fh:
        fh.writelines(data)


def get_data_folder(directory: Path) -> Path:
    """Get data folder from test directory"""
    root = directory.parent
    for folder in root.glob("data"):
        return folder


def get_file_map(source_dir: Path, target_dir: Path) -> Tuple[Path, Path]:
    """Make a file map from source to target file names"""
    test_file_names = [
        "test_portfolio.json",
        "test_profile.json",
        "test_transcript.json",
    ]
    file_names = [
        "portfolio.json",
        "profile.json",
        "transcript.json",
    ]
    return (
        [source_dir.joinpath(source), target_dir.joinpath(target)]
        for source, target in zip(file_names, test_file_names)
    )


def main() -> None:
    """Main entrypoint - generates assets for testing"""
    MAX_LINES = 10
    ROOT = Path(__file__).parent
    test_asset_folder = make_asset_folder(ROOT)
    data_folder = get_data_folder(ROOT)
    test_files = get_file_map(data_folder, test_asset_folder)
    for source, target in test_files:
        contents = read_truncated_file(source, MAX_LINES)
        write_file(target, contents)


if __name__ == "__main__":
    main()
