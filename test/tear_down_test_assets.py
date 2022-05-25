from pathlib import Path


def main():
    """Unlink Files in test/assets and remove assets directory"""
    expected_files = [
        "test_portfolio.json",
        "test_profile.json",
        "test_transcript.json",
    ]
    assets = Path(__file__).parent / "assets"
    for file in assets.glob("*.json"):
        if file.name in expected_files:
            file.unlink(missing_ok=True)
    if list(assets.glob("*")):
        raise FileExistsError(f"Files Exist in {assets}")
    assets.rmdir()


if __name__ == "__main__":
    main()
