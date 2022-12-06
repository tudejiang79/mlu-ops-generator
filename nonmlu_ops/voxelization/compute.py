import torch
import numpy as np
from nonmlu_ops.base import *
from mmcv.ops import Voxelization

@registerTensorList('voxelization')
class VoxelizationTensorList(TensorList):
    def random_generate(self, tensor_, tensor_idx, *args, **kwargs):
        if tensor_idx == 0:
            return super().random_generate(tensor_, tensor_idx, *args, **kwargs)
        elif tensor_idx == 1:
            return super().random_generate(tensor_, tensor_idx, *args, **kwargs)
        elif tensor_idx == 2:
            coors_range = tensor_.getDataNode().getData()
            while (coors_range[0] >= coors_range[3]) or (coors_range[1] >= coors_range[4]) or (coors_range[2] >= coors_range[5]):
                coors_range = tensor_.getDataNode().getData()
                super().random_generate(tensor_, tensor_idx, *args, **kwargs)
    # pass

@registerOp('voxelization')
class VoxelizationOp(OpTest):
    def __init__(self, tensor_list, params):
        print(torch.__version__)
        super().__init__(tensor_list, params)
        self.points_shape_ = self.tensor_list_.getInputTensor(0).getShape()
        self.voxel_size_shape_ = self.tensor_list_.getInputTensor(1).getShape()
        self.coors_range_shape_ = self.tensor_list_.getInputTensor(2).getShape()
        self.num_points = self.points_shape_[0]
        self.num_features = self.points_shape_[1]
        self.max_points = self.params_.get("max_points", 35)
        self.max_voxels = self.params_.get("max_voxels", 20000)
        self.NDim = self.params_.get("ndim", 3)
        self.deterministic = self.params_.get("deterministic", True)

    def computeOutputShape(self):
        self.tensor_list_.getOutputTensor(0).setShape([self.num_points, self.max_points, self.num_features])
        self.tensor_list_.getOutputTensor(1).setShape([self.num_points, 3])
        self.tensor_list_.getOutputTensor(2).setShape([self.num_points])
        self.tensor_list_.getOutputTensor(3).setShape([1])

    def compute(self):
        points_tensor = self.tensor_list_.getInputTensor(0)
        voxel_size_tensor = self.tensor_list_.getInputTensor(1)
        coors_range_tensor = self.tensor_list_.getInputTensor(2)
        voxels_tensor = self.tensor_list_.getOutputTensor(0)
        coors_tensor = self.tensor_list_.getOutputTensor(1)
        num_points_per_voxel_tensor = self.tensor_list_.getOutputTensor(2)
        voxel_num_tensor = self.tensor_list_.getOutputTensor(3)
        points = torch.tensor(points_tensor.getDataNode().getData()).contiguous().cuda()
        voxel_size = torch.tensor(voxel_size_tensor.getDataNode().getData())
        coors_range = torch.tensor(coors_range_tensor.getDataNode().getData())

        hard_voxelization = Voxelization(
            voxel_size.numpy().tolist(),
            coors_range.numpy().tolist(),
            self.max_points,
            self.max_voxels,
            self.deterministic)

        voxels_cut, coors_cut, num_points_per_voxel_cut = hard_voxelization.forward(points)
        voxel_num = torch.tensor([num_points_per_voxel_cut.size(0)])
        voxel_num_tensor.getDataNode().setData(voxel_num.numpy())
        voxels = points.new_zeros(size=(self.max_voxels, self.max_points, self.num_features))
        coors = points.new_zeros(size=(self.max_voxels, 3), dtype=torch.int)
        num_points_per_voxel = points.new_zeros(size=(self.max_voxels, ), dtype=torch.int)
        voxels = voxels + voxels_cut
        coors = coors + coors_cut
        num_points_per_voxel = num_points_per_voxel + num_points_per_voxel_cut
        voxels_tensor.getDataNode().setData(voxels.cpu().numpy())
        coors_tensor.getDataNode().setData(coors.cpu().numpy())
        num_points_per_voxel_tensor.getDataNode().setData(num_points_per_voxel.cpu().numpy())

@registerProtoWriter('voxelization')
class OpTensorProtoWriter(MluOpProtoWriter):
    def dumpOpParam2Node(self):
        param_node = self.proto_node_.voxelization_param
        param_node.max_points = self.op_params_.get("max_points", 35)
        param_node.max_voxels = self.op_params_.get("max_voxels", 20000)
        param_node.NDim = self.op_params_.get("ndim", 3)
        param_node.deterministic = self.op_params_.get("deterministic", True)
