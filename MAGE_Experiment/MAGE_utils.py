import os
from datasets import load_dataset

def process_dataset(dataset):
    def parse_src(example):
        """
        解析 src 字符串，将 src 划分为domain、model、model_family
        """
        parts = example['src'].split('_')

        # 修改拼写错误，gpt-3.5-trubo -> gpt-3.5-turbo
        if 'gpt-3.5-trubo' in parts:
            parts[parts.index('gpt-3.5-trubo')] = 'gpt-3.5-turbo'

        split_keys = ['machine', 'human', 'gpt4']
        split_idx = next((i for i, p in enumerate(parts) if p in split_keys), -1)

        # 划分model
        if parts[split_idx] == 'machine':
            model = '_'.join(parts[split_idx + 2:])
        else:
            model = '_'.join(parts[split_idx:])
        example['model'] = model

        # 定义模型到标签的映射字典
        model_mapping = {
            # 人类
            'human': 0,

            # OpenAI GPT 系列
            'gpt-3.5-turbo': 1,
            'text-davinci-002': 1,
            'text-davinci-003': 1,

            # LLaMA 系列
            '7B': 2,
            '13B': 2,
            '30B': 2,
            '65B': 2,

            # GLM
            'GLM130B': 3,

            # T5/FLAN 系列
            'flan_t5_small': 4,
            'flan_t5_base': 4,
            'flan_t5_large': 4,
            'flan_t5_xl': 4,
            'flan_t5_xxl': 4,

            # OPT 系列
            'opt_125m': 5,
            'opt_350m': 5,
            'opt_1.3b': 5,
            'opt_2.7b': 5,
            'opt_6.7b': 5,
            'opt_13b': 5,
            'opt_30b': 5,
            'opt_iml_max_1.3b': 5,
            'opt_iml_30b': 5,

            # BigScience系列
            'bloom_7b': 6,
            't0_3b': 6,
            't0_11b': 6,

            # EleutherAI系列
            'gpt_j': 7,
            'gpt_neox': 7
        }
        example['model_label'] = model_mapping.get(example['model'], -1)  # 使用-1表示未知模型

        return example

    for k, v in dataset.items():
        dataset[k] = v.map(parse_src)
    return dataset

def load_testbed1_data(base_path="../data/MAGE/domain_specific_model_specific"):
    """
    加载Testbed1数据集 (Fixed-domain & Model-specific)

    Args:
        base_path: 数据集根路径

    Returns:
        testbed1_datasets: 字典，key为领域名，value为对应的DatasetDict
    """

    # 定义所有领域
    domains = ['cmv', 'eli5', 'hswag', 'roct', 'sci_gen', 'squad', 'tldr', 'wp', 'xsum', 'yelp']

    testbed1_datasets = {}

    for domain in domains:
        domain_path = os.path.join(base_path, domain)

        # 检查路径是否存在
        if not os.path.exists(domain_path):
            print(f"警告: 路径 {domain_path} 不存在，跳过该领域")
            continue

        try:
            # 加载该领域的训练、验证、测试集
            dataset = load_dataset('csv',
                                 data_files={
                                     'train': os.path.join(domain_path, 'train.csv'),
                                     'validation': os.path.join(domain_path, 'valid.csv'),
                                     'test': os.path.join(domain_path, 'test.csv')
                                 })
            testbed1_datasets[domain] = dataset

        except Exception as e:
            print(f"加载领域 {domain} 时出错: {e}")
            continue

    return testbed1_datasets

def load_testbed2_data(base_path="../data/MAGE/cross_domains_model_specific"):
    """
    加载Testbed2数据集 (Arbitrary-domains & Model-specific)

    Args:
        base_path: 数据集根路径

    Returns:
        testbed2_datasets: 字典，key为模型集名称，value为对应的DatasetDict
    """

    # 定义模型集文件夹映射到论文中的模型集名称
    model_mapping = {
        'model__7B': 'LLaMA_set',
        'model_bloom_7b': 'BigScience_set',
        'model_flan_t5_small': 'FLAN_T5_set',
        'model_GLM130B': 'GLM_130B_set',
        'model_gpt-3.5-trubo': 'OpenAI_GPT_set',
        'model_gpt_j': 'EleutherAI_set',
        'model_opt_125m': 'OPT_set'
    }

    testbed2_datasets = {}

    for model_folder, model_set_name in model_mapping.items():
        model_path = os.path.join(base_path, model_folder)

        # 检查路径是否存在
        if not os.path.exists(model_path):
            print(f"警告: 路径 {model_path} 不存在，跳过该模型集")
            continue

        try:
            # 加载该模型集的训练、验证、测试集
            dataset = load_dataset('csv',
                                 data_files={
                                     'train': os.path.join(model_path, 'train.csv'),
                                     'validation': os.path.join(model_path, 'valid.csv'),
                                     'test': os.path.join(model_path, 'test.csv')
                                 })
            testbed2_datasets[model_set_name] = dataset

        except Exception as e:
            print(f"加载模型集 {model_set_name} (文件夹: {model_folder}) 时出错: {e}")
            continue

    return testbed2_datasets

def load_testbed3_data(base_path="../data/MAGE/domain_specific_cross_models"):
    """
    加载Testbed3数据集 (Fixed-domain & Arbitrary-models)

    Args:
        base_path: 数据集根路径

    Returns:
        testbed3_datasets: 字典，key为领域名称，value为对应的DatasetDict
    """

    # 定义所有领域
    domains = ['cmv', 'eli5', 'hswag', 'roct', 'sci_gen', 'squad', 'tldr', 'wp', 'xsum', 'yelp']

    testbed3_datasets = {}

    for domain in domains:
        domain_path = os.path.join(base_path, domain)

        # 检查路径是否存在
        if not os.path.exists(domain_path):
            print(f"警告: 路径 {domain_path} 不存在，跳过该领域")
            continue

        try:
            # 加载该领域的训练、验证、测试集
            dataset = load_dataset('csv',
                                 data_files={
                                     'train': os.path.join(domain_path, 'train.csv'),
                                     'validation': os.path.join(domain_path, 'valid.csv'),
                                     'test': os.path.join(domain_path, 'test.csv')
                                 })

            testbed3_datasets[domain] = dataset

        except Exception as e:
            print(f"加载领域 {domain} 时出错: {e}")
            continue

    return testbed3_datasets

def load_testbed4_data(base_path="../data/MAGE/cross_domains_cross_models"):
    """
    加载Testbed4数据集 (Arbitrary-domains & Arbitrary-models)

    Args:
        base_path: 数据集根路径

    Returns:
        testbed4_datasets: 包含训练、验证、测试集的DatasetDict
    """

    try:
        # 加载数据集
        testbed4_datasets = load_dataset('csv',
                             data_files={
                                 'train': os.path.join(base_path, 'train.csv'),
                                 'validation': os.path.join(base_path, 'valid.csv'),
                                 'test': os.path.join(base_path, 'test.csv')
                             })

    except Exception as e:
        print(f"加载Testbed4数据集时出错: {e}")
        raise

    return testbed4_datasets

def load_testbed5_data(base_path="../data/MAGE/unseen_models"):
    """
    加载Testbed5数据集 (Unseen Models)

    Args:
        base_path: 数据集根路径

    Returns:
        testbed5_datasets: 字典，key为排除的模型家族名称，value为对应的DatasetDict
    """

    # 定义模型文件夹到模型家族名称的映射
    model_mapping = {
        'unseen_model__7B': 'excluded_LLaMA_set',
        'unseen_model_bloom_7b': 'excluded_BigScience_set',
        'unseen_model_flan_t5_small': 'excluded_FLAN_T5_set',
        'unseen_model_GLM130B': 'excluded_GLM_130B_set',
        'unseen_model_gpt-3.5-trubo': 'excluded_OpenAI_GPT_set',
        'unseen_model_gpt_j': 'excluded_EleutherAI_set',
        'unseen_model_opt_125m': 'excluded_OPT_set'
    }

    testbed5_datasets = {}

    for model_folder, excluded_model_set in model_mapping.items():
        model_path = os.path.join(base_path, model_folder)

        # 检查路径是否存在
        if not os.path.exists(model_path):
            print(f"警告: 路径 {model_path} 不存在，跳过该模型集")
            continue

        try:
            # 加载该排除设置下的训练、验证、测试集
            dataset = load_dataset('csv',
                                 data_files={
                                     'train': os.path.join(model_path, 'train.csv'),
                                     'validation': os.path.join(model_path, 'valid.csv'),
                                     'test_id': os.path.join(model_path, 'test.csv'),  # in-distribution测试
                                     'test_ood': os.path.join(model_path, 'test_ood.csv')  # out-of-distribution测试
                                 })
            testbed5_datasets[excluded_model_set] = dataset

        except Exception as e:
            print(f"加载排除设置 {excluded_model_set} (文件夹: {model_folder}) 时出错: {e}")
            continue

    return testbed5_datasets

def load_testbed6_data(base_path="../data/MAGE/unseen_domains"):
    """
    加载Testbed6数据集 (Unseen Domains)

    Args:
        base_path: 数据集根路径

    Returns:
        testbed6_datasets: 字典，key为排除的领域名称，value为对应的DatasetDict
    """

    # 定义领域文件夹到领域名称的映射
    domain_mapping = {
        'unseen_domain_cmv': 'excluded_cmv',
        'unseen_domain_eli5': 'excluded_eli5',
        'unseen_domain_hswag': 'excluded_hswag',
        'unseen_domain_roct': 'excluded_roct',
        'unseen_domain_sci_gen': 'excluded_sci_gen',
        'unseen_domain_squad': 'excluded_squad',
        'unseen_domain_tldr': 'excluded_tldr',
        'unseen_domain_wp': 'excluded_wp',
        'unseen_domain_xsum': 'excluded_xsum',
        'unseen_domain_yelp': 'excluded_yelp'
    }

    testbed6_datasets = {}

    for domain_folder, excluded_domain in domain_mapping.items():
        domain_path = os.path.join(base_path, domain_folder)

        # 检查路径是否存在
        if not os.path.exists(domain_path):
            print(f"警告: 路径 {domain_path} 不存在，跳过该领域")
            continue

        try:
            # 加载该排除设置下的训练、验证、测试集
            dataset = load_dataset('csv',
                                 data_files={
                                     'train': os.path.join(domain_path, 'train.csv'),
                                     'validation': os.path.join(domain_path, 'valid.csv'),
                                     'test_id': os.path.join(domain_path, 'test.csv'),  # in-distribution测试
                                     'test_ood': os.path.join(domain_path, 'test_ood.csv')  # out-of-distribution测试
                                 })

            testbed6_datasets[excluded_domain] = dataset

        except Exception as e:
            print(f"加载排除设置 {excluded_domain} (文件夹: {domain_folder}) 时出错: {e}")
            continue

    return testbed6_datasets

def load_testbed7_data(base_path="../data/MAGE/test_ood_gpt.csv"):
    """
    加载Testbed7数据集 (Unseen Domains & Unseen Model)

    Args:
        base_path: Testbed7数据文件路径

    Returns:
        testbed7_datasets: 包含测试集的DatasetDict
    """

    try:
        # 加载CSV文件
        testbed7_datasets = load_dataset('csv', data_files={'test_ood_gpt': base_path})

    except Exception as e:
        print(f"加载Testbed7数据集时出错: {e}")
        raise

    return testbed7_datasets

def load_testbed8_data(base_path="../data/MAGE/test_ood_gpt_para.csv"):
    """
    加载Testbed8数据集 (Paraphrasing Attack)

    Args:
        base_path: Testbed8数据文件路径

    Returns:
        testbed8_datasets: 包含测试集的DatasetDict
    """

    try:
        # 加载CSV文件
        testbed8_datasets = load_dataset('csv', data_files={'test_ood_gpt_para': base_path})

    except Exception as e:
        print(f"加载Testbed8数据集时出错: {e}")
        raise

    return testbed8_datasets