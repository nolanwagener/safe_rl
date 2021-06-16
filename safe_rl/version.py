version_info = (0, 1, 0)
# format:
# ('safe_rl_major', 'safe_rl_minor', 'safe_rl_patch')

def get_version():
    "Returns the version as a human-format string."
    return '%d.%d.%d' % version_info

__version__ = get_version()
