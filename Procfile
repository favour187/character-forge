# 4 threads so the UI stays responsive during a build; the heavy pipeline is
# serialised in-process by memguard.slot(), so the threads can never stack
# several atlases on top of each other (that is what OOM-killed the box).
web: gunicorn -w 1 --threads 4 -t 180 -b 0.0.0.0:$PORT server:app
