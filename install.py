import sys

try:
    from launch import run_pip
except ImportError:
    import subprocess
    def run_pip(command: str, message: str):
        subprocess.check_call([sys.executable, "-m", "pip", command])
        print(message)

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
