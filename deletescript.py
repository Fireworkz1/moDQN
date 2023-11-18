import os

# 定义要删除的文件列表
files_to_delete = ['a.log', 'cost.txt','details.log','trainloss.txt','mdqn_model.ckpt','b.log']

# 获取当前目录
current_directory = os.getcwd()

# 构建文件路径并删除文件
for file_name in files_to_delete:
    file_path = os.path.join(current_directory, file_name)
    try:
        os.remove(file_path)
        print(f"{file_name} 删除成功")
    except OSError as e:
        print(f"删除 {file_name} 失败: {e}")
