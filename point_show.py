import numpy as np
import torch
# import open3d

def show(skeletion,predict):
    value = [300,1000,200]
    
    skeletion= skeletion.cpu()
    aa = skeletion[0,3:].reshape((-1,32,3)).numpy()
    skeletion_data = aa[0,:,:3]
    # trans_data = source_data[:]
    # mean = np.mean(source_data,axis=(0))
    for i in range (3):
        skeletion_data[:,i] *= value[i]
        
    predict= predict.detach().cpu()
    aa = predict[0,3:].reshape((-1,32,3)).numpy()
    predict_data = aa[0,:,:3]
    # trans_data = source_data[:]
    # mean = np.mean(source_data,axis=(0))
    for i in range (3):
         predict_data[:,i] *= value[i]
    point_cloud1 = open3d.geometry.PointCloud()
    point_cloud1.points = open3d.utility.Vector3dVector(skeletion_data)
    point_cloud1.paint_uniform_color((0,0,255))
    
    point_cloud2 = open3d.geometry.PointCloud()
    point_cloud2.points = open3d.utility.Vector3dVector(predict_data)
    point_cloud2.paint_uniform_color((0,255,0))
    
    open3d.visualization.draw_geometries([point_cloud1,point_cloud2],point_show_normal=True)
