import subprocess
import time

# 要运行的 Python 脚本文件
delete="delete_script.py"
train = "train.py"
pareto="pareto_sum.py"
subprocess.run(["python", delete])
for i in range(5):
    # 执行 Python 脚本
    subprocess.run(["python", train,str(i+1)])
    time.sleep(1)
subprocess.run(["python", pareto])
