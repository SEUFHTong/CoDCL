# import warnings
# from utils.DataLoader import load_or_create_data
# from utils.load_configs import get_link_prediction_args

# if __name__ == "__main__":

#     warnings.filterwarnings('ignore')

#     # get arguments
#     args = get_link_prediction_args(is_evaluation=False)

#     # get data for training, validation and testing
#     folder_path = './processed_data/my_dataset_folder/{}'.format(args.dataset_name)
#     node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
#     load_or_create_data(folder_path, dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)


import warnings
from utils.DataLoader import load_or_create_data
from utils.load_configs import get_link_prediction_args

def process_dataset(dataset_name):
    """处理单个数据集的完整流程"""
    warnings.filterwarnings('ignore')
    
    # 初始化参数并覆盖数据集名称
    args = get_link_prediction_args(is_evaluation=False)
    args.dataset_name = dataset_name  # 动态设置当前数据集
    
    try:
        print(f"\n{'='*40}\nProcessing dataset: {dataset_name}\n{'='*40}")
        
        # 构建数据路径
        folder_path = f'./processed_data/my_dataset_folder/{dataset_name}'
        
        # 加载或创建数据集
        data_tuple = load_or_create_data(
            folder_path=folder_path,
            dataset_name=dataset_name,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
        
        # 解包数据（根据实际返回结构调整）
        node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = data_tuple
        
        # 添加自定义处理逻辑
        print(f"成功加载 {dataset_name} 数据集")
        print(f"训练数据量: {len(full_data.src_node_ids)} 条")
        
        return True
    except Exception as e:
        print(f"处理数据集 {dataset_name} 时出错: {str(e)}")
        return False

if __name__ == "__main__":
    # 定义要处理的数据集列表
    datasets = [
        'wikipedia', 
        'mooc', 
        'enron', 
        'uci', 
        'CanParl', 
        'USLegis'
    ]
    
    # 执行多数据集处理
    success_count = 0
    for dataset in datasets:
        if process_dataset(dataset):
            success_count += 1
    
    # 输出汇总报告
    print(f"\n处理完成: 共 {len(datasets)} 个数据集")
    print(f"成功处理: {success_count} 个")
    print(f"失败处理: {len(datasets) - success_count} 个")