import subprocess
import yaml

sequence_paths = [
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_gray/dataset/sequences/00/',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_gray/dataset/sequences/01/',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_gray/dataset/sequences/02/',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_gray/dataset/sequences/03/',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_gray/dataset/sequences/04/',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_gray/dataset/sequences/05/',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_gray/dataset/sequences/06/',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_gray/dataset/sequences/07/',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_gray/dataset/sequences/08/',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_gray/dataset/sequences/09/',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_gray/dataset/sequences/10/',
]

gt_paths = [
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_poses/dataset/poses/00.txt',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_poses/dataset/poses/01.txt',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_poses/dataset/poses/02.txt',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_poses/dataset/poses/03.txt',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_poses/dataset/poses/04.txt',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_poses/dataset/poses/05.txt',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_poses/dataset/poses/06.txt',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_poses/dataset/poses/07.txt',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_poses/dataset/poses/08.txt',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_poses/dataset/poses/09.txt',
    '/media/ddc_robot/4cda377d-28db-4424-921c-6a1e0545ceeb/4cda377d-28db-4424-921c-6a1e0545ceeb/Dataset/kitti_odom/data_odometry_poses/dataset/poses/10.txt',
]

models = [
    'Alike',
    'SuperPoint',
    'XFeat',
    'DISK',
    'EdgePoint',
    'sfd2',
]

base_config = '/home/server/linyicheng/py_proj/keypoint_bench/keypoint_bench_git/keypoint_bench/config/config_vo.yaml'

for i in range(len(sequence_paths)):
    sequence_path = sequence_paths[i]
    gt_path = gt_paths[i]
    print('sequence_path:', sequence_path)

    for model in models:
        print('model:', model)

        with open(base_config, 'r') as file:
            config = yaml.safe_load(file)
        # 修改 model_type 的值
        config['test']['model']['params']['model_type'] = model
        config['test']['data']['params']['kitti_params']['root'] = sequence_path
        config['test']['data']['params']['kitti_params']['gt'] = gt_path
        # 将修改后的配置写回 YAML 文件
        with open("config.yaml", 'w') as file:
            yaml.safe_dump(config, file)

        result = subprocess.run(
            ['python', '/home/server/linyicheng/py_proj/keypoint_bench/keypoint_bench_git/keypoint_bench/main.py', '-c',
             'config.yaml', 'test'],
            capture_output=True, text=True)
        print(result.stdout)







