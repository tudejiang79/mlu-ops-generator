import torch
import numpy as np
from nonmlu_ops.base import *
from mmcv.ops import Voxelization

@registerTensorList('voxelization')
class VoxelizationTensorList(TensorList):
    pass

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
        self.NDim = self.params_.get("NDim", 3)

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

        hard_voxelization =  Voxelization(
            voxel_size.numpy().tolist(),
            coors_range.numpy().tolist(),
            self.max_points,
            self.max_voxels,
            deterministic=True)
        points = torch.tensor(points).contiguous().to(device='cuda:0')
        voxels, coors, num_points_per_voxel = hard_voxelization.forward(points)
        voxels_tensor.getDataNode().setData(voxels.cpu().numpy())
        coors_tensor.getDataNode().setData(coors.cpu().numpy())
        num_points_per_voxel_tensor.getDataNode().setData(num_points_per_voxel.cpu().numpy())

@registerProtoWriter('voxelization')
class OpTensorProtoWriter(MluOpProtoWriter):
    def dumpOpParam2Node(self):
        param_node = self.proto_node_.voxelization_param
        param_node.max_points = self.op_params_.get("max_points", 35)
        param_node.max_voxels = self.op_params_.get("max_voxels", 20000)
        param_node.NDim = self.op_params_.get("NDim", 3)
