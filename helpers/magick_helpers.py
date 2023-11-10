

def get_magic_command(command, magick_executible_override_path = None):
    if magick_executible_override_path is None:
        return f'magick {command}'
    
    return f'{magick_executible_override_path} {command}'