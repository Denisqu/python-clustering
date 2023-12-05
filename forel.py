# https://github.com/tsolakghukasyan/FOREL-clustering/blob/master/cluster.py
# https://logic.pdmi.ras.ru/~sergey/teaching/ml/11-cluster.pdf (слайды 29-30)
# http://www.machinelearning.ru/wiki/index.php?title=Алгоритм_ФОРЕЛЬ

from scipy import spatial
import numpy as np
from sklearn import datasets
from utils import myplot
import copy, random

class FOREL:
    '''
    На каждом шаге мы случайным образом выбираем объект из выборки, раздуваем вокруг него сферу радиуса R,
    внутри этой сферы выбираем центр тяжести и делаем его центром новой сферы.
    Т.о. мы на каждом шаге двигаем сферу в сторону локального сгущения объектов выбоки,
    т.е. стараемся захватить как можно больше объектов выборки сферой фиксированного радиуса.
    После того как центр сферы стабилизируется, все объекты внутри 
    сферы с этим центром мы помечаем как кластеризованные и выкидываем их из выборки.
    Этот процесс мы повторяем до тех пор, пока вся выборка не будет кластеризована.
    '''
    def __init__(self, r=2):
        self.r = r
        self.clusters = []
    
    def __call__(self, initial_arr):
        arr = copy.deepcopy(initial_arr)
        self.__cluster(arr)
        return self.clusters

    def __cluster(self, arr):
        while not self.__is_finished(arr):
            current_object = self.__get_random_non_clustered_object(arr) # Случайно выбираем текущий объект из выборки
            print(f"__cluster, current_object = {current_object}")
            close_objects_arr = self.__get_close_objects(arr, current_object) # Помечаем объекты выборки, находящиеся на расстоянии менее, чем R от текущего
            centeral_object = self.__central_object(close_objects_arr) # Вычисляем их центр тяжести, помечаем этот центр как новый текущий объект

            while not (centeral_object[0] == current_object[0] and centeral_object[1] == current_object[1]): #Повторяем шаги 2-3, пока новый текущий объект не совпадет с прежним
                current_object = centeral_object
                print(f"__cluster, current_object = {current_object}")
                close_objects_arr = self.__get_close_objects(arr, current_object)
                centeral_object = self.__central_object(close_objects_arr)
                print(f"current object = {current_object}, central_object = {centeral_object}")
            

            self.clusters.append(close_objects_arr)
            #arr = np.setdiff1d(arr, close_objects_arr) # Помечаем объекты внутри сферы радиуса R вокруг текущего объекта как кластеризованные, выкидываем их из выборки
            self.__remove_objects_from_array(arr, close_objects_arr)
            print(f"new arr length = {len(arr)}")
        
        if len(arr) == 1:
            self.clusters.append(close_objects_arr)


    def __is_finished(self, arr) -> bool:
        print(f"__is_finished, arr len = {len(arr)}")
        return len(arr) <= 1
    
    def __get_random_non_clustered_object(self, arr):
        return arr[random.randrange(0, len(arr))]
    
    def __get_close_objects(self, arr, current_object):
        print(f"__get_close_objects, arr = {arr}, current_object = {current_object}")
        return [point for point in arr if np.linalg.norm(current_object - point) < self.r]
   
    
    def __central_object(self, objects_arr):
        obj_0 = 0
        obj_1 = 0
        print(f"__central_object, objects_arr = {objects_arr}, objects_arr[0] = {obj_0}, objects_arr[1] = {obj_1}")
        #https://stackoverflow.com/questions/15819980/calculate-mean-across-dimension-in-a-2d-array
        mean_coords = np.mean(objects_arr, axis=0)
        #find closesest to mean_coords:
        # https://stackoverflow.com/questions/10818546/optimize-finding-index-of-nearest-point-in-2d-arrays
        nearest_point = objects_arr[spatial.KDTree(objects_arr).query(mean_coords)[1]]

        #mean_y = np.mean(objects_arr, axis=1)
        # TODO: какие-то странные значения выводит в консоль, надо разобраться
        print(f"mean_coords = {mean_coords}, nearest_point = {nearest_point}")
        #return np.array([mean_x, mean_y])
        return nearest_point
    
    def __remove_objects_from_array(self, array, objects):
        print(f"ARRAY LEN: {len(array)}")
        for array_item_index in range(len(array) - 1, 0, -1):
            print(f"array_item_index = {array_item_index}")
            removed_object = None
            for object in objects:
                print(f"array_item = {array[array_item_index][0]}")
                if object[0] == array[array_item_index][0] and object[1] == array[array_item_index][1]:
                    removed_object = object
                    break
            print(f"deleting array_item = {array_item_index}")
            # TODO: сейчас как-то неправильно удаляется
            array = np.delete(array, array_item_index)
            print(f"new size of array = {len(array)}")
            
        
if __name__ == '__main__':
    # creating a dataset for clustering
    X, y = datasets.make_blobs()
    print(f"X.shape = {X.shape}, y.shape = {y.shape}")
    print(X)
    forel = FOREL(1)
    result = forel(X)
    
    for i, cluster in enumerate(result):
        print(f"{i}, {cluster}")
    #y_preds = run_Kmeans(3, X)
    #Getting unique labels
    #p = myplot.Plot()
    #p.plot_in_2d(X, y_preds, title="K-Means Clustering")
    #p.plot_in_2d(X, y, title="Actual Clustering")