import os,signal,time
from pathlib import Path
victims=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\0')
  env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x)
 except (OSError,ProcessLookupError):continue
 if args and b'python' in args[0] and b'tool/simulator/ros_planning_web.py' in args and env.get(b'TINYNAV_EXPERIMENT_MODE') in (b'baseline',b'model'):
  victims.append(int(p.name));os.kill(int(p.name),signal.SIGTERM);print('Stopped experimental server',p.name,flush=True)
end=time.monotonic()+8
while time.monotonic()<end and any(Path('/proc',str(pid)).exists() for pid in victims):time.sleep(.1)
