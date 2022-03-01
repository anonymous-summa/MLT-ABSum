import os

path = "model/"

mode = "cnndm-pg"
path = path+mode+"/"
files = os.listdir(path)
files = sorted(files)[-3:]
print(files)

for file_name in files:
    if "cnndm" in mode:
        os.system(f"""
            python -u predict.py --left 0 --right 1500 --cuda cuda:1 --model_name {path+file_name} --mode {mode} & sleep 20
            python -u predict.py --left 1500 --right 3000 --cuda cuda:1 --model_name {path+file_name} --mode {mode} &
            python -u predict.py --left 3000 --right 4500 --cuda cuda:1 --model_name {path+file_name} --mode {mode} & 
            python -u predict.py --left 4500 --right 6000 --cuda cuda:0 --model_name {path+file_name} --mode {mode} &
            python -u predict.py --left 6000 --right 7500 --cuda cuda:0 --model_name {path+file_name} --mode {mode} &
            python -u predict.py --left 7500 --right 9000 --cuda cuda:0 --model_name {path+file_name} --mode {mode} &
            python -u predict.py --left 9000 --right 10500 --cuda cuda:0 --model_name {path+file_name} --mode {mode} &
            python -u predict.py --left 10500 --right 12000 --cuda cuda:1 --model_name {path+file_name} --mode {mode} &
            wait
            python -u eval.py --model_name {path+file_name} --mode {mode}
            wait
            """
        )

