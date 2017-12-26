import re
import shlex
from subprocess import Popen, PIPE, STDOUT


runtimes = []

for i in range(50, 350, 50):
  
  job = 'python3 -m ai.tests.qalb --model_name=gpu_test_{0} --batch_size={0}'
  
  p = Popen(shlex.split(job.format(i)), stdout=PIPE, stderr=STDOUT)
  
  expected_step = 0
  step_runtimes = []
  last_line = ''
  
  for line in p.stdout:
    last_line = line.rstrip().decode('utf-8') 
    print(last_line)
    
    m = re.match(r'Global step {} \(([0-9]+)s'.format(expected_step), last_line)
    if m:
      step_runtimes.append(float(m.group(1)))
      expected_step += 1
      if expected_step == 10:
        p.kill()
        break
  
  point = (i, min(step_runtimes))
  runtimes.append(point)

print('-' * 80)
for point in runtimes:
  print(point)
