import subprocess


def get_git_full():
    lines = subprocess.check_output(['git', 'log', '-n', '1']).decode('utf-8').split('\n')
    return [s.strip() for s in lines if len(s.strip()) > 1]


def get_git_hash():
    return subprocess.check_output(
        ['git', 'log', '-n', '1', '--pretty=tformat:%H']
    ).strip().decode('utf-8')


def get_git_short_hash():
    return subprocess.check_output(
        ['git', 'log', '-n', '1', '--pretty=tformat:%h']
    ).strip().decode('utf-8')


def get_git_short_hash_and_commit_date():
    return subprocess.check_output(
        ['git', 'log', '-n', '1', '--pretty=tformat:%h-%ad', '--date=short']
    ).strip().decode('utf-8')
