import yaml
import subprocess

# 模型列表
model_names = ["Alike", "SuperPoint", "XFeat", "D2Net", "DISK", "r2d2", "sfd2", "EdgePoint", ]


# 读取并修改config.yaml文件中的模型类型
def update_model_type(model_name):
    # 加载配置文件
    with open("config/config_AUC.yaml", "r") as file:
        config = yaml.safe_load(file)

    # 更新模型类型
    config["test"]["model"]["params"]["model_type"] = model_name

    # 保存修改后的配置文件
    with open("config/config_run.yaml", "w") as file:
        yaml.dump(config, file)

# 运行python命令
def run_command():
    subprocess.run(["python3", "main.py", "-c", "./config/config_run.yaml", "test"])

# 主函数
def main():
    for model_name in model_names:
        print(f"Running with model: {model_name}")
        update_model_type(model_name)  # 更新配置文件
        run_command()  # 运行命令

if __name__ == "__main__":
    main()

