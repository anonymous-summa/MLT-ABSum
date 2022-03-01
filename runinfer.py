import os

op = "train"

os.system(
    f"""
        python -u inference.py --left 30000 --right 40000 --cuda cpu --op {op} &
        python -u inference.py --left 40000 --right 50000 --cuda cpu --op {op} &
        python -u inference.py --left 50000 --right 60000 --cuda cpu --op {op} &
        wait
    """)



