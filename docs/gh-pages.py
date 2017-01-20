import subprocess

subprocess.call(['make', 'clean', 'html',
                 # 'latexpdf'
                 ])
subprocess.call(['ghp-import', '-p', '-n', 'build/html'])
