import subprocess

subprocess.call(['make', 'clean', 'html'])
subprocess.call(['ghp-import', '-p', '-n', '_build/html'])
