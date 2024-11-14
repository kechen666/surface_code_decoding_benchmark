# author: Kevin Chen
import stim

import logging
# 设置 logging 配置
# logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

from typing import List, Union, Optional, Tuple, Set, Dict
import numpy as np

class MaxLikelihoodDecoder:
    def __init__(self, detector_error_model: stim.DetectorErrorModel, detector_number: Union[int, None] = None, logical_number: Union[int, None] = 1):
        """构建最大似然解码器的初始化

        Args:
            detector_error_model (stim.DetectorErrorModel): 检测错误模型
            detector_number (Union[int, None], optional): 检测器的个数. Defaults to None.
            logical_number (Union[int, None], optional): 逻辑比特的个数. Defaults to 1.
        """
        
        self.detector_error_model = detector_error_model
        if detector_number is None:
            self.detector_number = detector_error_model.num_detectors
        else:
            self.detector_number = detector_number
        
        if logical_number is None:
            self.logical_number = detector_error_model.num_observables
        else:
            self.logical_number = logical_number
        
        self.detector_error_model_dict = self.get_detector_error_model_dict(detector_error_model, self.detector_number)
        
    def decode(self, measurement_outcomes: List[str]) -> List[int]:
        """解码操作

        Args:
            measurement_outcomes (List[str]): 用于解码的syndrome, 对应测量比特的结果, 格式为['01101010', '10010101'].

        Returns:
            List[int]: 是否进行逻辑纠错, 其中1表示进行纠错, 0表示不进行纠错。
        """
        error_correction_operation = []
        for syndrome in measurement_outcomes:
            # 计算在当前检测器观测值下，对应的错误分布。
            syndrome_probability_distribution = self.compute_syndrome_probability_distribution(syndrome)
            
            # 找到当前检测器观测值下，概率最大的逻辑比特观测值（即是否发生逻辑错误，是否需要纠正）
            max_probability_detector_observable =  max(syndrome_probability_distribution, key=syndrome_probability_distribution.get)
            probability = syndrome_probability_distribution[max_probability_detector_observable] / sum(syndrome_probability_distribution.values())
            observable = max_probability_detector_observable[self.detector_number:]
            
            if len(observable) == 1:
                # 只有一个量子比特被翻转
                if observable == "0":
                    logger.info(f"no error, no use logical flip, the correct probability rate is {probability}")
                    error_correction_operation.append(0)
                else:
                    logger.info(f"error, use logical flip, the correct probability rate is {probability}")
                    error_correction_operation.append(0)
            else:
                raise("Now, only for one logical qubit")
        return error_correction_operation

    def get_detector_logical_observable_val(self, targets_copy: List[stim.DemTarget], detector_number: int) -> List[int]:
        """从一个error事件中, 获取它所翻转的detector和logical_observable的index

        Args:
            targets_copy (List[stim.DemTarget]): 错误事件中的detector或logical_observable对象
            detector_number (int): detector的个数
            
        Returns:
            List[int]: 翻转的detector和logical_observable的index列表
        """
        target_val = []
        for target in targets_copy:
            if target.is_relative_detector_id():
                target_val.append(target.val)
            elif target.is_logical_observable_id():
                target_val.append(target.val + detector_number)
        return target_val
    
    def get_detector_error_model_dict(self, detector_error_model: stim.DetectorErrorModel, detector_number:int) -> Dict[Tuple[int], float]:
        """生成一个detector和logical_observable的翻转index的作为key, 出现概率作为value的字典

        Args:
            detector_error_model (stim.DetectorErrorModel): 错误检测模型
            detector_number (int): detector的个数

        Returns:
            Dict[Tuple[int], float]: detector和logical_observable的翻转index的作为key, 出现概率作为value的字典. 
            eg. {(0,2): 0.000005,.....,(7,):0.00005}
        """
        detector_error_model_dict = {}

        for error in detector_error_model:
            if error.type == "error":
                probability = error.args_copy()[0]
                targets_copy = error.targets_copy()
                fliped_detector_observable_index = self.get_detector_logical_observable_val(targets_copy, detector_number)
                detector_error_model_dict[tuple(fliped_detector_observable_index)] = probability
        return detector_error_model_dict 
                    
    def flip_detector_observable_index_by_flip_index(self, detector_observable: str, flip_detector_observable_index: Tuple[int]) -> str:
        """根据翻转的检测器或逻辑观测值的index, 翻转检测器或逻辑观测值

        Args:
            detector_observable (str): 检测器或逻辑观测值
            flip_detector_observable_index (Tuple[int]): 翻转的检测器或逻辑观测值的index

        Returns:
            str: 翻转后的检测器或逻辑观测值
        """
        fliped_detector_observable = detector_observable
    
        for i in flip_detector_observable_index:
            if detector_observable[i] == '0':
                fliped_detector_observable = fliped_detector_observable[:i]+'1'+fliped_detector_observable[i+1:]
            else:
                fliped_detector_observable = fliped_detector_observable[:i]+'0'+fliped_detector_observable[i+1:]
        
        return fliped_detector_observable
    
    def compute_syndrome_probability_distribution(self, syndrome: str) -> Dict[str, float]:
        """当前syndrome(测量比特的测量值)的概率分布，具体的计算方法如下：
        1. 初始化一个字典, key为全为0表示detector+observable的字符串, value为概率, 初始值为1.
        2. 遍历syndrome中的每一个detector_i, 作用与detector_i相关的所有错误事件, 更新字典中的概率分布.（每个错误事件作用一次, 我们根据最前面的detector来取相关错误事件, 这样每个错误事件就只取一次）
        3. 在作用完与当前detector_i的所有错误事件后, 根据detector_i的测量值, 删除一些无关的分布。例如detector_i测量值为0, 那么所有与detector_i为1的分布将不需要考虑。

        Args:
            syndrome (str): 测量比特的测量值

        Returns:
            Dict[str, float]: 当前syndrome(测量比特的测量值)的概率分布, 规模与逻辑比特数量有关, 对于逻辑比特为n, 一般为2^n。
        """
        if isinstance(syndrome, str):
            pass
        else:
            raise TypeError("syndrome must be a string")
        
        error_probability_distribution = {}
        initial_detector_observable = '0'*(self.detector_number + self.logical_number)
        error_probability_distribution[initial_detector_observable] = 1
        
        for detector_i in range(self.detector_number):
            syndrome_detector_i = syndrome[detector_i]
            # 找到第i个检测器对应的所有错误事件，这里采用的是k[0]，因为默认detector_error_model_dict是存在顺序的，k[0]一般指的是最小的检测器编号
            related_detector_error_model_dict = {k:v for k,v in self.detector_error_model_dict.items() if detector_i == k[0]}
            for flip_detector_observable_index, flip_probability in related_detector_error_model_dict.items():
                # 执行与第i个检测器相关的所有错误事件。
                probability_distribution = {}
                for detector_observable, detector_observable_probability in error_probability_distribution.items():
                    # no flip
                    no_fliped_detector_observable = detector_observable
                    no_fliped_probability = detector_observable_probability * (1-flip_probability)
                    # flip
                    fliped_detector_observable = self.flip_detector_observable_index_by_flip_index(detector_observable, flip_detector_observable_index)
                    fliped_probability = detector_observable_probability * flip_probability
                    
                    # 在同一个错误事件中，可能出现相同的检测器、逻辑观测值，则概率相加
                    probability_distribution[no_fliped_detector_observable] = probability_distribution.get(no_fliped_detector_observable, 0) + no_fliped_probability
                    probability_distribution[fliped_detector_observable] = probability_distribution.get(fliped_detector_observable, 0) + fliped_probability
                # 更新错误概率分布
                error_probability_distribution = probability_distribution
            # 根据检测器i的值，缩小错误概率分布的范围
            error_probability_distribution = {k:v for k,v in error_probability_distribution.items() if syndrome_detector_i == k[detector_i]}
            # 输入在线MLD方法，按照detector的顺序执行，每步执行的规模。
            logger.debug(f"detector index: {detector_i}, uncomputed and connected edge number: {len(related_detector_error_model_dict)}, probability distribution length: {len(error_probability_distribution)}")
        return error_probability_distribution
    
    def compute_correcation_error_logical_probability(self, error_syndromes: np.array) -> float:
        """计算当前这些syndrome解码出错, 可能导致逻辑错误增加的概率。
           目前暂时只支持单个逻辑比特的判断。

        Args:
            error_syndromes (np.array): 出错的syndrome

        Returns:
            float: 总概率。即所有syndrome解码出错, 可能导致逻辑错误概率。
        """
        error_probability = 0
        
        for syndrome in error_syndromes:
            syndrome_probability_distribution = self.compute_syndrome_probability_distribution(syndrome)
            print(f"error syndrome_probability_distribution:{syndrome_probability_distribution}")
            max_probability_detector_observable =  max(syndrome_probability_distribution, key=syndrome_probability_distribution.get)
            probability = 2 * syndrome_probability_distribution[max_probability_detector_observable] - sum(syndrome_probability_distribution.values())
            error_probability += probability
        
        return error_probability
    
    def compute_correcation_logical_probability(self, error_syndromes: np.array) -> float:
        """计算当前这些syndrome解码都对了,对应的逻辑错误率是多少？
           目前暂时只支持单个逻辑比特的判断。

        Args:
            error_syndromes (np.array): 出错的syndrome

        Returns:
            float: 总概率。即所有syndrome解码出错, 可能导致逻辑错误概率。
        """
        logical_error_probability = 0
        
        for syndrome in error_syndromes:
            syndrome_probability_distribution = self.compute_syndrome_probability_distribution(syndrome)
            max_probability_detector_observable =  max(syndrome_probability_distribution, key=syndrome_probability_distribution.get)
            probability = sum(syndrome_probability_distribution.values()) - syndrome_probability_distribution[max_probability_detector_observable]
            logical_error_probability += probability
        
        return logical_error_probability
    
    def compute_detector_number_distribution(self) -> Dict[str, int]:
        """计算在detector error model中, 每个detector包含的错误机制数量, 在超图中表示每个节点的度。

        Returns:
            Dict[str, int]: 通过字典表示每个detector包含的错误机制数量。{"D0":19, "D1":12,..., "L0":xxx}
        """
        detector_number_distribution = {}
        for error in self.detector_error_model:
            if error.type == "error":
                targets_copy = error.targets_copy()
                fliped_detector_observable_index = self.get_detector_logical_observable_val(targets_copy, self.detector_number)
                for index in fliped_detector_observable_index:
                    key: str = ""
                    if index >= self.detector_number:
                        key = f"L{index - self.detector_number}"
                    else:
                        key = f"D{index}"
                    detector_number_distribution[key] = detector_number_distribution.get(key, 0) + 1
        return detector_number_distribution
    
    def compute_detector_connectivity(self, have_logical_observable: bool = True) -> Dict[str, List[int]]:
        
        """计算在detector error model中, 每个detector相邻的detector数量, 在超图中不等于节点的度。
        
        Args:
            have_logical_observable (bool, optional): 是否包含逻辑态对应的检测器. Defaults to True. 如果为False, 即不包括L0项, D_i项中也不包含逻辑态对应的检测器索引。
        
        Returns:
            Dict[str, List[int]]: 通过字典表示每个detector相邻的detector。{"D0":[1,2,...],"D1":{0,2,...},...,"L0":[,...,]}
        """
        detector_connectivity = {}
        for error in self.detector_error_model:
            if error.type == "error":
                targets_copy = error.targets_copy()
                fliped_detector_observable_index = self.get_detector_logical_observable_val(targets_copy, self.detector_number)
                for index in fliped_detector_observable_index:
                    fliped_detector_observable_index_copy = fliped_detector_observable_index.copy()
                    key: str = ""
                    if index >= self.detector_number:
                        if have_logical_observable:
                            # 如果包含逻辑态对应的检测器，则将逻辑态对应的检测器索引减去检测器数量
                            key = f"L{index - self.detector_number}"
                        else:
                            # 如果不包含逻辑态对应的检测器，则跳过
                            continue
                    else:
                        key = f"D{index}"
                    fliped_detector_observable_index_copy.remove(index)
                    for detector in fliped_detector_observable_index_copy:
                        # 如果检测器索引大于等于检测器数量，则表示是逻辑态对应的检测器，直接跳过
                        if detector >= self.detector_number:
                            if have_logical_observable:
                                pass
                            else:
                                continue
                        value = detector_connectivity.get(key, [])
                        if detector not in detector_connectivity.get(key, []):
                            value.append(detector)
                            detector_connectivity[key] = value
        return detector_connectivity

    def get_connected_detector_coordinates(self, detector_connectivity: Dict[str, List[int]], detector_index: int) -> List[Tuple[int, int]]:
        """获取与指定detector相邻的detector的坐标
        
        Args:
            detector_connectivity (Dict[str, List[int]]): 每个detector相邻的detector
            detector_index (int): 指定的detector的index
            detector_error_model (DetectorErrorModel): detector error model
            
        Returns:
            List[Tuple[int, int]]: 相邻的detector的坐标 
        """
        connected_detector_coordinates = []
        detector_name = f"D{detector_index}"
        connected_detector_indexs = detector_connectivity[detector_name]
        detector_coordinates = self.detector_error_model.get_detector_coordinates()
        
        for detector_index in connected_detector_indexs:
            if detector_index<self.detector_number:
                # 如果是detector，则直接获取坐标，否则是logical observable
                detector_coordinate = detector_coordinates[detector_index]
                connected_detector_coordinates.append(detector_coordinate)
        return connected_detector_coordinates