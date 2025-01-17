# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
import json
import argparse
from plyfile import PlyData, PlyElement

parser = argparse.ArgumentParser()
parser.add_argument('--fractaldb_path', default='./PC-FractalDB/3Dfractalmodel', help='load PLY path')
parser.add_argument('--save_dir', default='./PC-FractalDB/3Dfractalscene', help='save JSON path')
parser.add_argument('--numof_scene', type=int, default=10000, help='the number of 3D fractal scene')
parser.add_argument('--numof_classes', type=int, default=1000, help='the number of fractal category')
parser.add_argument('--numof_instance', type=int, default=1000, help='the number of intra-category')
parser.add_argument('--numof_object', type=int, default=50, help='the average object per 3D fractal scene')
parser.add_argument('--scene_size', type=float, default=15.0, help='the size of 3D fractal scene')
FLAGS = parser.parse_args()

src_dir = FLAGS.fractaldb_path
NUM_CLASS = FLAGS.numof_classes
NUM_INS = FLAGS.numof_instance
NUM_OBJ = FLAGS.numof_object
ROOM_SIZE = FLAGS.scene_size
dump_dir = FLAGS.save_dir
num_scenes = FLAGS.numof_scene
# 地形フラクタル生成関数 (ランダムパラメータ対応)
def generate_random_terrain():
    size = np.random.choice([129, 257, 513])  # 地形のグリッドサイズ（ランダム選択）
    scale = np.random.uniform(2.0, 10.0)      # 高さスケール（ランダム選択）
    terrain = generate_terrain(size, scale)
    return terrain, size, scale

# 地物の密度をランダム化する関数
def setRandomObjNum(avg_obj):
    return avg_obj + int(np.random.poisson(lam=np.random.uniform(3, 10)))
# シーンを生成する関数の修正
def createOneSceneWithDensity(NUM_OBJ, terrain, room_size):
    avg_obj = setObjNum(NUM_OBJ)  # 平均のオブジェクト数を設定
    num_obj = setRandomObjNum(avg_obj)  # ランダムな密度でオブジェクト数を設定
    objects = []

    for obj_id in range(num_obj):
        box = setOneObject(objects, terrain, room_size)
        if box is None:
            break
        
        obj = {}
        obj["size"] = box["size"].tolist()
        obj["c_pos"] = box["c_pos"].tolist()
        obj["theta"] = (1.0 - np.random.random()*2.0) * np.pi
        file_path = src_files[np.random.randint(0, len(src_files))]
        obj["class_name"] = os.path.basename(os.path.dirname(file_path))
        obj["file_name"] = os.path.basename(file_path)
        objects.append(obj)
    return objects

def setObjNum(NUM_OBJ):
    return NUM_OBJ + int(np.floor(np.random.poisson(lam=5)))

def setObjSize():
    base_size = 0.75 + np.random.random()*0.5   # uniform random in [1.00, 5.00]
    aspects = np.ones(3, dtype=np.float32)
    aspects[1:3] = 0.9 + np.random.random(2) * 0.2  # uniform random in [0.9, 1.1]
    return aspects * base_size

def _checkInterpositionBox(box1, box2):
    def checkInterpos(p1, p2):
        return p1[0] > p2[1] and p2[0] > p1[1]

    def getEdge(box, axis):
        c = box["c_pos"][axis]
        w = box["size"][axis] * 0.5
        return [ c+w, c-w ]

    for axis in [0,1]:
        if not checkInterpos(getEdge(box1, axis), getEdge(box2, axis)):
            return False
    return True

def _checkInterpositionBox2(box1, box2):
    c1 = np.array(box1["c_pos"][:2])
    c2 = np.array(box2["c_pos"][:2])
    s1 = np.array(box1["size"][:2])
    s2 = np.array(box2["size"][:2])
    return np.linalg.norm(c1-c2) < (np.linalg.norm(s1) + np.linalg.norm(s2)) * 0.5 + 0.25
# 地形フラクタル点群を生成
def generate_terrain_point_cloud(terrain, room_size, resolution=0.1):
    points = []
    size = terrain.shape[0]
    scale = room_size / (size - 1)
    for i in range(size):
        for j in range(size):
            x = (i - size // 2) * scale
            y = (j - size // 2) * scale
            z = terrain[i, j]
            points.append([x, y, z])
    return np.array(points, dtype=np.float32)

# load_object_point_cloud関数の修正
def load_object_point_cloud(obj, src_dir, room_size, class_idx):
    file_path = os.path.join(src_dir, obj["class_name"], obj["file_name"])
    if not os.path.exists(file_path):
        return None

    # PLYファイルの読み込み
    plydata = PlyData.read(file_path)
    points = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T

    # サイズ変更
    points *= np.array(obj["size"])

    # 回転 (Z軸回りの回転)
    theta = obj["theta"]
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    points = points @ rotation_matrix.T

    # 平行移動
    points += np.array(obj["c_pos"])

    return points  # クラスIDは呼び出し側で追加する

# 点群をPLYファイルとして保存
def save_point_cloud_to_ply(file_path, points):
    vertices = [(p[0], p[1], p[2], int(p[3])) for p in points]
    vertex_element = PlyElement.describe(
        np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('class_id', 'i4')]), 'vertex')
    PlyData([vertex_element], text=True).write(file_path)
# 地形フラクタル生成関数 (ダイヤモンド-スクエア法)
def generate_terrain(size, scale):
    terrain = np.zeros((size, size))
    terrain[0, 0] = np.random.random() * scale
    terrain[0, -1] = np.random.random() * scale
    terrain[-1, 0] = np.random.random() * scale
    terrain[-1, -1] = np.random.random() * scale
    
    step_size = size - 1
    while step_size > 1:
        half_step = step_size // 2

        # ダイヤモンドステップ
        for x in range(0, size - 1, step_size):
            for y in range(0, size - 1, step_size):
                avg = (terrain[x, y] +
                       terrain[x + step_size, y] +
                       terrain[x, y + step_size] +
                       terrain[x + step_size, y + step_size]) / 4.0
                terrain[x + half_step, y + half_step] = avg + (np.random.random() - 0.5) * scale

        # スクエアステップ
        for x in range(0, size, half_step):
            for y in range((x + half_step) % step_size, size, step_size):
                avg = (terrain[(x - half_step) % (size - 1), y] +
                       terrain[(x + half_step) % (size - 1), y] +
                       terrain[x, (y - half_step) % (size - 1)] +
                       terrain[x, (y + half_step) % (size - 1)]) / 4.0
                terrain[x, y] = avg + (np.random.random() - 0.5) * scale

        step_size //= 2
        scale /= 2.0

    return terrain

# 地形上の高さを取得する関数
def get_height_from_terrain(terrain, x, y, room_size):
    size = terrain.shape[0]
    grid_x = int((x / room_size + 0.5) * (size - 1))
    grid_y = int((y / room_size + 0.5) * (size - 1))
    return terrain[grid_x, grid_y]

# 修正した setOneObject 関数
def setOneObject(objects, terrain, room_size):
    s_box = setObjSize()
    c_z = s_box[2] * 0.5

    ITERATION_MAX = 100
    for itr in range(ITERATION_MAX):
        c_pos = np.zeros(3, dtype=np.float32)
        c_pos[:2] = (1.0 - np.random.random(2)*2.0) * room_size * 0.5  # [-ROOM_SIZE/2, ROOM_SIZE/2]
        c_pos[2] = get_height_from_terrain(terrain, c_pos[0], c_pos[1], room_size) + c_z
        box = {"size": s_box, "c_pos": c_pos}

        accept_flg = True
        for obj in objects:
            if _checkInterpositionBox2(box, obj):
                accept_flg = False
                break
        if accept_flg:
            return box
    return None


def createOneScene(NUM_OBJ):
    num_obj = setObjNum(NUM_OBJ)
    objects = []
    for obj_id in range(num_obj):
        box = setOneObject(objects)
        if box is None:
            break
        
        obj = {}
        obj["size"] = box["size"].tolist()
        obj["c_pos"] = box["c_pos"].tolist()
        obj["theta"] = (1.0 - np.random.random()*2.0) * np.pi
        file_path = src_files[np.random.randint(0, len(src_files))]
        obj["class_name"] = os.path.basename(os.path.dirname(file_path))
        obj["file_name"] = os.path.basename(file_path)
        objects.append(obj)
    return objects

if __name__ == '__main__':
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    src_files = []
    for class_idx in range(NUM_CLASS):
        c_files = glob.glob(os.path.join(src_dir, "%06d" % class_idx, "*.ply"))
        c_files.sort()
        src_files.extend(c_files[0:NUM_INS])

    print(len(src_files))

    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
        print("create", dump_dir)

    for idx in range(num_scenes):
        print("\r%d/%d" % (idx + 1, num_scenes), end='')

        # ランダムな地形生成
        terrain, terrain_size, terrain_scale = generate_random_terrain()

        # 地形点群生成
        terrain_points = generate_terrain_point_cloud(terrain, ROOM_SIZE)

        # 地形点群にクラスID列を追加（-1でクラス番号を示す）
        terrain_class_ids = np.full((terrain_points.shape[0], 1), -1, dtype=np.int32)
        scene_points = np.hstack([terrain_points, terrain_class_ids])

        # シーンファイル名の設定
        file_name = os.path.join(dump_dir, "scene_%05d.json" % idx)
        ply_file_name = os.path.join(dump_dir, "scene_%05d.ply" % idx)

        # シーンに含まれるオブジェクト情報を生成
        objects = createOneSceneWithDensity(NUM_OBJ, terrain, ROOM_SIZE)

        # オブジェクトの点群を統合
        for obj in objects:
            class_idx = int(obj["class_name"])  # クラス番号を取得
            obj_points = load_object_point_cloud(obj, src_dir, ROOM_SIZE, class_idx)

            if obj_points is not None:
                # オブジェクト点群にクラスID列を追加
                obj_class_ids = np.full((obj_points.shape[0], 1), class_idx, dtype=np.int32)
                obj_points_with_class = np.hstack([obj_points, obj_class_ids])
                
                # シーン点群に追加
                scene_points = np.vstack([scene_points, obj_points_with_class])

        # 点群をPLYファイルとして保存
        save_point_cloud_to_ply(ply_file_name, scene_points)

        # シーン情報をJSONファイルとして保存
        with open(file_name, 'w') as f:
            json.dump(objects, f, indent=4)

    print('')
  
