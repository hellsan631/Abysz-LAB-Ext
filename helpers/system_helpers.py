import subprocess

def run_system_command(command, job_path = None):
    try:
        print(f"Running command: {command} and {job_path}")
        subprocess.check_call(command, cwd=job_path, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error with command: {command} and {job_path}")
        raise e