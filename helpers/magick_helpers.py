import os

def get_magic_command(command, magick_executible_override_path = None):
    if magick_executible_override_path is None or magick_executible_override_path.trim() == '':
        return f'magick {command}'
    
    # ensure that `magick_executible_override_path` actually exists
    if not os.path.exists(magick_executible_override_path):
        raise FileNotFoundError(f'Could not find ImageMagick executable at {magick_executible_override_path}')
    
    # ensure that `magick_executible_override_path` is properly escaped for windows
    # so that the future `subprocess.check_call` does not fail, as the file path might include spaces
    if os.name == 'nt':
        magick_executible_override_path = f'"{magick_executible_override_path}"'
    return f'{magick_executible_override_path} {command}'