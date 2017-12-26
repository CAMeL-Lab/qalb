"""Batch size runtime testing for GPUs. This script takes a number of steps and
   a step size, and will output the runtimes for the QALB test. This way one can
   check for which batch sizes does the running time increase less than in
   direct proportionality."""

import re
import shlex
from subprocess import Popen, PIPE, STDOUT
import sys


try:
  NUM_STEPS = int(sys.argv[1])
  STEP_SIZE = int(sys.argv[2])

except IndexError:
  print("Usage: python gpu_test.py [num_steps] [step_size]")
  exit()


runtimes = []

for i in range(STEP_SIZE, (NUM_STEPS + 1) * STEP_SIZE, STEP_SIZE):
  
  job = 'python3 -m ai.tests.qalb --model_name=gpu_test_{0} --batch_size={0}'
  
  p = Popen(shlex.split(job.format(i)), stdout=PIPE, stderr=STDOUT)
  
  expected_step = 0
  step_runtimes = []
  last_line = ''
  
  for line in p.stdout:
    last_line = line.rstrip().decode('utf-8') 
    print(last_line)
    
    m = re.match(r'Global step (?:[0-9]+) \(([0-9\.]+)s', last_line)
    if m:
      step_runtimes.append(float(m.group(1)))
      expected_step += 1
      if expected_step >= 10:
        print("Stopping...")
        p.kill()
        break
  
  point = (i, min(step_runtimes))
  runtimes.append(point)

print('-' * 80)
for point in runtimes:
  print(point)
