import subprocess, os, sys
src = os.path.join('pyht', 'solver', '_solver_c.c')
out = os.path.join('pyht', 'solver', '_solver_c.dll')
cmd = ['gcc', '-O3', '-march=native', '-shared', '-o', out, src, '-lm']
print('Compiling: ' + ' '.join(cmd))
ret = subprocess.run(cmd, cwd=os.path.dirname(__file__) or '.').returncode
if ret == 0:
    print('OK: ' + out)
else:
    print('FAILED (gcc not found or compilation error)')
    sys.exit(ret)
