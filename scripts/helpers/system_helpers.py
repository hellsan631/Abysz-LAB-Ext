import subprocess

def run_system_command(command, job_path = None):
    return subprocess.check_call(command, cwd=job_path, shell=True)