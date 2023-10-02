import launch

from pathlib import Path


req_file = Path(__file__).parent / "requirements.txt"

with req_file.open() as file:
    for package in file:
        package_name = package.strip()
        try:
            launch.run_pip(f"install -U {package_name}", f"sd-webui-prompt-forge requirement: {package_name}")
        except Exception as e:
            print(e)
            print(f"Failed to install {package_name}, script prompt-forge might not work")
