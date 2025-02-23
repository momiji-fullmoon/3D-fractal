# -*- coding: utf-8 -*-
import argparse
import os
import random
import time
import numpy as np
import open3d
import open3d
import numpy as np
import os
import random

class ifs_function():
    def __init__(self):
        self.prev_x,self.prev_y, self.prev_z = 0.0, 0.0, 0.0
        self.function  = []
        self.xs,self.ys, self.zs = [],[],[]
        self.select_function = []
        self.temp_proba = 0.0

    def set_param(self,a,b,c,d,e,f,g,h,i,j,k,l,proba, **kwargs):
        if "weight_a" in kwargs:
            a *= kwargs["weight_a"]
        if "weight_b" in kwargs:
            b *= kwargs["weight_b"]
        if "weight_c" in kwargs:
            c *= kwargs["weight_c"]
        if "weight_d" in kwargs:
            d *= kwargs["weight_d"]
        if "weight_e" in kwargs:
            e *= kwargs["weight_e"]
        if "weight_f" in kwargs:
            f *= kwargs["weight_f"]
        if "weight_g" in kwargs:
            g *= kwargs["weight_g"]
        if "weight_h" in kwargs:
            h *= kwargs["weight_h"]
        if "weight_i" in kwargs:
            i *= kwargs["weight_i"]
        if "weight_j" in kwargs:
            j *= kwargs["weight_j"]
        if "weight_k" in kwargs:
            k *= kwargs["weight_k"]
        if "weight_l" in kwargs:
            l *= kwargs["weight_l"]
        temp_function  = {"a":a,"b":b,"c":c,"d":d,"e":e,"f":f,"g":g,"h":h,"i":i,"j":j,"k":k,"l":l,"proba":proba}
        self.function.append(temp_function)
        self.temp_proba += proba
        self.select_function.append(self.temp_proba)


    def calculate(self,iteration):
        """ Recursively calculate coordinates for args.iteration """
        rand = np.random.random(iteration)
        select_function = self.select_function
        function = self.function
        prev_x, prev_y, prev_z = self.prev_x, self.prev_y, self.prev_z
        for i in range(iteration-1):
            for j in range(len(select_function)):
                if rand[i] <= select_function[j]:
                    next_x = prev_x*function[j]["a"] + \
                            prev_y*function[j]["b"] + \
                            prev_z*function[j]["c"] + \
                            function[j]["j"]
                    next_y = prev_x*function[j]["d"] + \
                            prev_y*function[j]["e"] + \
                            prev_z*function[j]["f"] + \
                            function[j]["k"]
                    next_z = prev_x*function[j]["g"] + \
                            prev_y*function[j]["h"] + \
                            prev_z*function[j]["i"] + \
                            function[j]["l"]
                    break
            self.xs.append(next_x), self.ys.append(next_y), self.zs.append(next_z)
            prev_x, prev_y, prev_z = next_x, next_y, next_z
        point_data = np.array((self.xs,self.ys,self.zs), dtype = float)
        return point_data
def conf():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", default="./PC-FractalDB/3DIFS_param",
                        type=str, help="path to csv file save directory")
    parser.add_argument("--numof_classes", default=10000, type=int, help="PC FractalDB category number")
    parser.add_argument("--start_numof_classes", default=0, type=int, help="")
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--point_num", default=10000, type=int, help="PointCloud number per one 3D Fractal model")
    parser.add_argument("--variance", default=0.05, type=float)
    parser.add_argument("--normalize", default=1.0, type=float)
    args = parser.parse_args()
    return args

def generator(args, params):
    generators = ifs_function()
    for param in params:
        generators.set_param(float(param[0]), float(param[1]),
                             float(param[2]), float(param[3]),
                             float(param[4]), float(param[5]),
                             float(param[6]), float(param[7]),
                             float(param[8]), float(param[9]),
                             float(param[10]), float(param[11]),
                             float(param[12]))
    data = generators.calculate(args.point_num)
    return data

def min_max(args, x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = ((x-min)/(max-min)) * (args.normalize - (-args.normalize)) - args.normalize
    return result

def getPcScale(point_cloud):
    scale_x = np.max(point_cloud[:,0]) - np.min(point_cloud[:,0])
    scale_y = np.max(point_cloud[:,1]) - np.min(point_cloud[:,1])
    scale_z = np.max(point_cloud[:,2]) - np.min(point_cloud[:,2])
    return max(max(scale_x, scale_y), scale_z)

def centoroid(point):
    new_centor = []
    sum_x = (sum(point[0]) / args.point_num)
    sum_y = (sum(point[1]) / args.point_num)
    sum_z = (sum(point[2]) / args.point_num)
    centor_of_gravity = [sum_x, sum_y, sum_z]
    fractal_point_x = (point[0] - centor_of_gravity[0]).tolist()
    fractal_point_y = (point[1] - centor_of_gravity[1]).tolist()
    fractal_point_z = (point[2] - centor_of_gravity[2]).tolist()
    new_centor.append(fractal_point_x)
    new_centor.append(fractal_point_y)
    new_centor.append(fractal_point_z)
    new = np.array(new_centor)
    return new

if __name__ == "__main__":
    start_time = time.time()
    args = conf()
    print(args)
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    os.makedirs(args.save_root, exist_ok=True)

    while(args.start_numof_classes < args.numof_classes):
        param_size = np.random.randint(2, 8)
        params = np.zeros((param_size, 13), dtype=float)
        sum_proba = 0.0
        
        for m in range(param_size):
            param_rand = np.random.uniform(-1.0, 1.0, 12)
            a, b, c, d, e, f, g, h, i, j, k, l = param_rand[0:12]
            check_param = np.array(param_rand[0:9], dtype=float).reshape(3, 3)
            prob = abs(np.linalg.det(check_param))
            # prob = abs((a*e*i)+(b*f*g)+(c*d*h)-(c*e*g)-(a*f*h)-(b*d*i))
            sum_proba += prob
            params[m, 0:13] = a, b, c, d, e, f, g, h, i, j, k, l, prob

        for m in range(param_size):
            params[m, 12] /= sum_proba
        
        fractal_point = generator(args, params)

        # min-max normalize
        point = min_max(args, fractal_point, axis=None)
        # move to center point 
        point = centoroid(point)
        # calucurate variance
        var_point = np.var(point, axis=1)
        # search N/A value
        arr = np.isnan(point).any(axis=1)
        if arr[1] == False:
            if var_point[0] > args.variance:
                if var_point[1] > args.variance:
                    if var_point[2] > args.variance:
                        point_data = point.transpose()
                        pointcloud = open3d.geometry.PointCloud()
                        pointcloud.points = open3d.utility.Vector3dVector(point_data)
                        # fractal_point_pcd = open3d.visualization.draw_geometries([pointcloud])
                        class_str = '%06d' % args.start_numof_classes
                        save = [class_str, param_size]
                        np.savetxt("{}/{}.csv".format(args.save_root, class_str), params, delimiter=",")
                        args.start_numof_classes += 1
    end_time = time.time()
    interval_time = end_time - start_time
    print("elapsed time = %dh %dm %ds" % (int(interval_time / 3600),
                                          int((interval_time % 3600) / 60),
                                          int((interval_time % 3600) % 60)))