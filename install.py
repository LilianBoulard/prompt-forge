import sys

from launch import run_pip

packages = {
    ("jsonschema", True),
    ("toml", sys.version_info.major == 3 and sys.version_info.minor <= 10)  # Not required from 3.11 onward
}


for package, required in packages:
    if required:
        try:
            run_pip(f"install -U {package}", f"sd-webui-prompt-forge requirement: {package}")
        except Exception as e:
            print(e)
            print(f"Failed to install {package}, script prompt-forge might not work")
