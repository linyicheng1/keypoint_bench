import subprocess
import yaml

models = [
    # 'XFeat',
    # 'EdgePoint',
    'sfd2',
    # 'Alike',
    'DISK',
    # 'SuperPoint',
]

base_config = '/home/server/linyicheng/py_proj/keypoint_bench/keypoint_bench_git/keypoint_bench/config/config_AUC.yaml'


for model in models:
    print('Model:', model)
    with open(base_config, 'r') as file:
        config = yaml.safe_load(file)
    # 修改 model_type 的值
    config['test']['model']['params']['model_type'] = model
    output = config['test']['model']['params']['AUC_params']['output']
    # 将修改后的配置写回 YAML 文件
    with open("config.yaml", 'w') as file:
        yaml.safe_dump(config, file)
    result = subprocess.run(['python', '/home/server/linyicheng/py_proj/keypoint_bench/keypoint_bench_git/keypoint_bench/main.py', '-c', 'config.yaml', 'test'],
                            capture_output=True, text=True)
    print(result.stdout)

    # create dir if not exist
    subprocess.run(['mkdir', '-p', output + model])
    # move matches* files to output dir
    subprocess.run(['mv '+output+'matches* '+output + model], shell=True)


