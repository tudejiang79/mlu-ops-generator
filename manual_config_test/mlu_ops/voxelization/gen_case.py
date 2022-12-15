import sys
import numpy as np
from allpairspy import AllPairs


def getShape(shapes):
    shape_val = '"shape":['
    for i in range(len(shapes) - 1):
        shape_val += str(shapes[i]) + ','
    shape_val += str(shapes[len(shapes) - 1]) + ']'
    return shape_val

def getType(data_type):
    return '"dtype":"' + data_type + '"'

def getOnchipType(data_type):
    return '"onchip_dtype":"' + data_type + '"'

def getRandomDistribution(distribution, start, end):
    return '"random_distribution":{"' + distribution + '":[' + str(start) + ',' + str(end) + ']}'

def getLayout(layout):
    return '"layout":"' + layout + '"'

def getStr(src_list):
    str_temp = "["
    for i in range(len(src_list)):
        str_temp += str(src_list[i])
        if i != len(src_list) - 1 :
            str_temp += ","
    str_temp += "]"
    return str_temp


def getSingleCase(num_points,num_features,max_voxels,max_points):
    points = [num_points, num_features]
    voxel_size = [3]
    coors_range = [6]
    voxels = [max_voxels, max_points, num_features]
    coors = [max_voxels, 3]
    num_points_per_voxel = [max_voxels]
    voxel_num = [1]

    data_type = "float32"

    points_str = '    {"inputs":[ \n    {' + getShape(points) + ',' + getType(data_type) + ',' + \
                  getRandomDistribution("uniform",100, 100) + ',' + getLayout("ARRAY") + ',' + \
                  '"contain_nan":false, "contain_inf":false},\n'
    
    voxel_size_str = '    {' + getShape(voxel_size) + ',' + getType(data_type) + ',' + \
                   getRandomDistribution("uniform",1, 100) + ',' + getLayout("ARRAY") + ',' + \
                  '"contain_nan":false, "contain_inf":false},\n'
    coors_range_str = '    {' + getShape(coors_range) + ',' + getType(data_type) + ',' + \
                   getRandomDistribution("uniform",100, 100) + ',' + getLayout("ARRAY") + ',' + \
                   '"contain_nan":false, "contain_inf":false}],\n' 
    
    voxels_str = '    "outputs":[ ' + \
                '{' + getShape(voxels) + ',' + getType(data_type) + ',' + \
                   getLayout("ARRAY") + '},\n'
    coors_str = '    {' + getShape(coors) + ',' + getType("int32") + ',' + \
                   getLayout("ARRAY") + '},\n'
    num_points_per_voxel_str = '    {' + getShape(coors) + ',' + getType("int32") + ',' + \
                   getLayout("ARRAY") + '},\n'              
    voxel_num_str = '    {' +getShape(voxel_num) + ',' + getType("int32") + ',' + \
                   getLayout("ARRAY") + '}],\n' 
    op_params = '    "op_params":{"max_points":'+str(max_points) +', "max_voxels":'+str(max_voxels) +',"ndim":3, "deterministic": true},\n'
    proto_params = '    "proto_params":{"write_data":true}}'

    cur_res = points_str + voxel_size_str + coors_range_str + voxels_str + coors_str + num_points_per_voxel_str + voxel_num_str + op_params + proto_params
    return cur_res


def genCase():
    num_points,num_features,max_voxels,max_points = 1,1,1,1
    parameters = [
        [1,5,128,1024],
        [1,8,255,1024],
        [1,16,224,1228],
        [1,32,512,2048]
        ]
    cur_res = '     "manual_data" : [\n'
    cur_res += getSingleCase(num_points,num_features,max_voxels,max_points)
    for i,params in enumerate(AllPairs(parameters)):
        cur_res += ',\n' + getSingleCase(params[0],params[1],params[2],params[3])
    cur_res += '\n ] \n}'
    return cur_res


if __name__ == "__main__":
    
    for i in range(1):
        res = '{\n\
        "op_name": "voxelization",\n\
        "device": "gpu",\n\
        "random_distribution": {"uniform":[-100,100]},\n\
        "supported_mlu_platform":["370", "590"],\n\
        "require_value":true,\n\
        "if_dynamic_threshold": false,\n\
        "evaluation_criterion": ["diff3"],\n\
        "evaluation_threshold":[0],\n'
    
        res += genCase()
        file_name = 'voxelization_manual.json'
        file = open(file_name, "w")
        file.write(res)
        file.close()