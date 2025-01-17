# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np

import open3d
import random

import open3d
import numpy as np
import os
import random
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

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
	parser.add_argument("--load_root", default="./PC-FractalDB/3DIFS_param", type = str, help="load csv root")
	parser.add_argument("--save_root", default="./PC-FractalDB/3Dfractalmodel", type = str, help="save PLY root")
	parser.add_argument("--iteration", default=10000, type = int)
	parser.add_argument("--numof_classes", default=1000, type = int)
	parser.add_argument("--start_class", default=0, type = int)
	parser.add_argument("--numof_instance", default=1000, type = int)
	parser.add_argument("--normalize", default=1.0, type=float)
	parser.add_argument("--ratio", default=0.8, type=float)
	parser.add_argument('--visualize', action='store_true')
	args = parser.parse_args()
	return args

def min_max(args, x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = ((x-min)/(max-min)) * (args.normalize - (-args.normalize)) - args.normalize
    return result

def centoroid(point):
	new_centor = []
	sum_x = (sum(point[0]) / args.iteration)
	sum_y = (sum(point[1]) / args.iteration)
	sum_z = (sum(point[2]) / args.iteration)
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
	starttime = time.time()
	args = conf()

	# object_list = ['object', 'main', 'mix']
	# for i in object_list:
	os.makedirs(args.save_root, exist_ok=True)

	csv_names = os.listdir(args.load_root)
	csv_names.sort()
	mix_csv_names = os.listdir(args.load_root)
	mix_csv_names = random.sample(mix_csv_names, k=args.numof_instance)

	for i, csv_name in enumerate(csv_names):
		name, ext = os.path.splitext(str(csv_name))
		name = args.start_class + int(name)
		name = '%06d' % name

		if i > args.numof_classes:
			break

		if ext != ".csv":
			continue

		os.makedirs(os.path.join(args.save_root, name), exist_ok=True)

		params = np.genfromtxt(args.load_root + "/" + csv_name, dtype=np.str_, delimiter=",")
		main_generators = ifs_function()
		main_obj_num = args.iteration * args.ratio
			
		for param in params:
			main_generators.set_param(float(param[0]), float(param[1]),float(param[2]), float(param[3]),
				float(param[4]), float(param[5]),float(param[6]), float(param[7]),
				float(param[8]), float(param[9]),float(param[10]), float(param[11]), float(param[12]))

		main_fractal_point = main_generators.calculate(int(main_obj_num))
		main_fractal_point = min_max(args, main_fractal_point, axis=None)
		main_fractal_point = centoroid(main_fractal_point)
		main_point_data = main_fractal_point.transpose()
		main_pointcloud = open3d.geometry.PointCloud()
		main_pointcloud.points = open3d.utility.Vector3dVector(main_point_data)

		fractal_weight = 0
		for j, mix_csv in enumerate(mix_csv_names):
			padded_fractal_weight= '%04d' % fractal_weight
			mix_generators = ifs_function()
			if j == args.numof_instance:
				break
			mix_params = np.genfromtxt(args.load_root + "/" + mix_csv, dtype=np.str_, delimiter=",")
			for mix_param in mix_params:
					mix_generators.set_param(float(mix_param[0]), float(mix_param[1]),float(mix_param[2]), float(mix_param[3]),
					float(mix_param[4]), float(mix_param[5]),float(mix_param[6]), float(mix_param[7]),
					float(mix_param[8]), float(mix_param[9]),float(mix_param[10]), float(mix_param[11]), float(mix_param[12]))
			mix_obj_num = args.iteration * (1 - args.ratio)
			mix_fractal_point = mix_generators.calculate(int(mix_obj_num) + 1)
			mix_fractal_point = min_max(args, mix_fractal_point, axis=None)
			mix_fractal_point = centoroid(mix_fractal_point)
			mix_point_data = mix_fractal_point.transpose()
			mix_pointcloud = open3d.geometry.PointCloud()
			mix_pointcloud.points = open3d.utility.Vector3dVector(mix_point_data)
			fractal_point = np.concatenate((main_fractal_point, mix_fractal_point), axis = 1) 

			# min-max normalize
			# fractal_point = min_max(args, fractal_point, axis=None)
			# move to center point 
			fractal_point = centoroid(fractal_point)
			# search N/A value
			arr = np.isnan(fractal_point).any(axis=1)

			if arr[1] == False:
				point_data = fractal_point.transpose()
				pointcloud = open3d.geometry.PointCloud()
				pointcloud.points = open3d.utility.Vector3dVector(point_data)
				if args.visualize == True:
					fractal_point_pcd = open3d.visualization.draw_geometries([pointcloud])
				open3d.io.write_point_cloud((args.save_root + "/" + name + "/" + name + "_" + padded_fractal_weight + ".ply"), pointcloud)
			fractal_weight += 1
	
	endtime = time.time()
	interval = endtime - starttime
	print("passed time = %dh %dm %ds" % (int(interval/3600),int((interval%3600)/60),int((interval%3600)%60)))
