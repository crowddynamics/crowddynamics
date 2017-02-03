import subprocess

subprocess.call(['make', 'clean', 'html',
                 # 'latexpdf'
                 ])
subprocess.call(['ghp-import', '-p', '-n', '_build/html'])
