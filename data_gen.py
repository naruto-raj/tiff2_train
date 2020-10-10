import os
import time
import gdal
import base64
import json
from pyproj import Proj, transform
from skimage.io import imread,imsave
import lxml.etree as etree
import cv2
from osgeo import gdal,osr
import rasterio
import rasterio.mask
from shapely.geometry import Polygon, Point
import geopandas as gpd

class data_generator():

    def __init__(self,tifffile, vector_list, outputdir,window_W=1000,window_H=1000,step_size=500, boundary=None,out_type='json'):
        self.input_tiff = tifffile
        self.input_vector = vector_list
        self.winW = int(window_W)
        self.winH = int(window_H)
        self.steps = int(step_size)
        self.bound = boundary
        self.outputDir = outputdir
        self.out_type = out_type
        self.objects={}
        for vector in os.listdir(self.input_vector):
            self.objects[vector[:-4]] = []
        self.img = imread(self.input_tiff)
        image = gdal.Open(self.input_tiff)
        self.geoTrans = image.GetGeoTransform()
        print("Image Loaded Successfully")
        self.read_all_vectors()
        self.label_gen()


    def Pixel2world(self,geoMatrix, x, y):
        ulX = geoMatrix[0]
        ulY = geoMatrix[3]
        xDist = geoMatrix[1]
        yDist = geoMatrix[5]
        return (ulX + (x * xDist)), (ulY + (y * yDist))

    def world2Pixel(self,geoMatrix, x, y):
      ulX = geoMatrix[0]
      ulY = geoMatrix[3]
      xDist = geoMatrix[1]
      yDist = geoMatrix[5]
      return (x - ulX) / xDist, (y - ulY) / yDist


    def vector_reader(self,vector_file_path,geotrans,type=None):
        print(vector_file_path)
        vector_read = gpd.read_file(vector_file_path, driver='ESRI Shapefile')
        final_coords = []
        for i in vector_read.geometry:
            outer_coords = i.exterior.coords.xy
            feature_coord = []
            for j in range(len(outer_coords[0])):
                feature_coord.append((outer_coords[0][j], outer_coords[1][j]))
            final_coords.append(feature_coord)
            new_polygon = []
            for polygon in final_coords:
                new_coord = []
                for coord in polygon:
                    new_coord.append(list(self.world2Pixel(self.geoTrans, coord[0], coord[1])))
                new_polygon.append(new_coord)
        return new_polygon

    def read_all_vectors(self):
        for file_name in os.listdir(self.input_vector):
            if file_name.endswith(".shp"):
                vector_file_path = os.path.join(self.input_vector, file_name)
                file_name = file_name.split('/')[-1][:-4]
                self.objects[file_name] = self.vector_reader(vector_file_path,self.geoTrans)

    def midpoint(self,p1, p2):
        return Point((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    def check_collision_point(self,bound, detected):
        polygon = Polygon(bound)
        return polygon.contains(detected)

    def sliding_window(self,image, stepSize, windowSize):
        # slide a window across the image/
        for y in range(0, image.shape[0], stepSize):
            for x in range(0, image.shape[1], stepSize):
                # yield the current window
                print(x,y,windowSize)
                yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

    def crop_tiff(self):
        bound_read = gpd.read_file(self.bound, driver='ESRI Shapefile')
        feature = bound_read['geometry']
        with rasterio.open(self.input_tiff,"r") as src:
            out_image, out_transform = rasterio.mask.mask(src, feature,crop=True)
            out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})
        with rasterio.open(self.bound[:-4]+".tif", "w", **out_meta) as dest:
            dest.write(out_image)
        return self.bound[:-4]+".tif"

    def create_object(self,annotation,nam,kml,ind):
        objectg = etree.SubElement(annotation, 'object')
        name = etree.SubElement(objectg, 'name')
        name.text = nam
        pose = etree.SubElement(objectg, 'pose')
        pose.text = 'Unspecified'
        truncated = etree.SubElement(objectg, 'truncated')
        truncated.text = '0'
        difficult = etree.SubElement(objectg, 'difficult')
        difficult.text = '0'
        bndbox = etree.SubElement(objectg, 'bndbox')
        xmin = etree.SubElement(bndbox, 'xmin')
        xmin.text = str(kml[nam][ind][0][0])
        ymin = etree.SubElement(bndbox, 'ymin')
        ymin.text = str(kml[nam][ind][0][1])
        xmax = etree.SubElement(bndbox, 'xmax')
        xmax.text = str(kml[nam][ind][1][0])
        ymax = etree.SubElement(bndbox, 'ymax')
        ymax.text = str(kml[nam][ind][1][1])

    def create_xml_file(self,folder_name, file_name, img_path_dir, kml, xrect):
        folder_n = folder_name
        filenameg = file_name
        path_n = img_path_dir + '/' + folder_n + '/' + filenameg + '.jpg'
        s_width = str(xrect[2] - xrect[0])
        s_height = str(xrect[3] - xrect[1])
        s_depth = '3'

        annotation = etree.Element('annotation')
        folder = etree.SubElement(annotation, 'folder')
        folder.text = folder_n
        filename = etree.SubElement(annotation, 'filename')
        filename.text = filenameg
        path = etree.SubElement(annotation, 'path')
        path.text = path_n
        source = etree.SubElement(annotation, 'source')
        database = etree.SubElement(source, 'database')
        database.text = 'Unknown'
        size = etree.SubElement(annotation, 'size')
        width = etree.SubElement(size, 'width')
        width.text = s_width
        height = etree.SubElement(size, 'height')
        height.text = s_height
        depth = etree.SubElement(size, 'depth')
        depth.text = s_depth
        segmented = etree.SubElement(annotation, 'segmented')
        for key in kml.keys():
            no_of_points = len(kml[key])
            if no_of_points>0:
                for j in range(0, no_of_points):
                    self.create_object(annotation=annotation, nam=key, kml=kml,ind = j)

        mydata = etree.tostring(annotation, pretty_print=True)
        myfile = open(img_path_dir + '/' + folder_n + '/' + filenameg + '.xml', 'wb')
        myfile.write(mydata)

    def create_json_file(self,folder_name, file_name, img_path_dir, kml, xrect):
        folder_n = folder_name
        filenameg = file_name
        path_n = img_path_dir + '/' + '/' + filenameg + '.jpg'
        with open(path_n, mode='rb') as file:
            img = file.read()
        imagedata= base64.b64encode(img).decode("utf-8")
        s_width = int(xrect[2] - xrect[0])
        s_height = int(xrect[3] - xrect[1])
        json_initial = {
            "version": "4.2.9",
            "flags": {},
            "shapes":[],
            "imagePath": path_n,
            "imageData":imagedata,
            "imageHeight":s_height,
            "imageWidth": s_width,
        }
        for key in kml.keys():
            no_of_points = len(kml[key])
            if no_of_points>0:
                for j in range(0, no_of_points):
                    shape_data = {"label": key, "points": kml[key][j], "shape_type": "polygon","flags": {}}
                    json_initial['shapes'].append(shape_data)
        myfile = open(img_path_dir + '/' + folder_n + '/' + filenameg + '.json', 'w')
        json.dump(json_initial,myfile)

    def label_gen(self):
        i = 0
        if self.bound != None:
            result = self.crop_tiff()
            self.img = imread(result)
        for (x, y, window) in self.sliding_window(self.img, stepSize=self.steps, windowSize=(self.winW, self.winH)):
            flag = 0
            i += 1
            filter_objects = {}
            for label in os.listdir(self.input_vector):
                filter_objects[label[:-4]] = []
            image_bound = [(x, y), (x, y + self.winH), (x + self.winW, y + self.winH), (x + self.winW, y)]
            xrect = [x, y, x + self.winH, y + self.winW]
            for key in self.objects.keys():
                points = []
                for point in self.objects[key]:
                    if self.out_type =='xml':
                        if self.check_collision_point(image_bound, Point([point[0][0], point[0][1]])):
                            if self.check_collision_point(image_bound, Point([point[2][0], point[2][1]])):
                                crop_img = window.copy()
                                x1_p = point[0][0]
                                y1_p = point[0][1]
                                x2_p = point[2][0]
                                y2_p = point[2][1]
                                point = [(x1_p - x, y1_p - y), (x2_p - x, y2_p - y)]
                                points.append(point)
                            elif x + 2000 - point[2][0] > -15 and x + 2000 - point[2][0] < 0:
                                point[2][0] = x + 2000 - 1
                                if self.check_collision_point(image_bound, Point([point[2][0], point[2][1]])):
                                    crop_img = window.copy()
                                    x1_p = point[0][0]
                                    y1_p = point[0][1]
                                    x2_p = point[2][0]
                                    y2_p = point[2][1]
                                    point = [(x1_p - x, y1_p - y), (x2_p - x, y2_p - y)]
                                    points.append(point)
                            elif y + 2000 - point[2][1] > -15 and y + 2000 - point[2][1] < 0:
                                point[2][1] = y + 2000 - 1
                                if self.check_collision_point(image_bound, Point([point[2][0], point[2][1]])):
                                    crop_img = window.copy()
                                    x1_p = point[0][0]
                                    y1_p = point[0][1]
                                    x2_p = point[2][0]
                                    y2_p = point[2][1]
                                    point = [(x1_p - x, y1_p - y), (x2_p - x, y2_p - y)]
                        elif x - point[0][0] < 15 and x - point[0][0] > 0:
                            point[0][0] = x + 1
                            if self.check_collision_point(image_bound, Point([point[2][0], point[2][1]])):
                                crop_img = window.copy()
                                x1_p = point[0][0]
                                y1_p = point[0][1]
                                x2_p = point[2][0]
                                y2_p = point[2][1]
                                point = [(x1_p - x, y1_p - y), (x2_p - x, y2_p - y)]
                                points.append(point)
                        elif y - point[0][1] < 15 and y - point[0][1] > 0:
                            point[0][1] = y + 1
                            if self.check_collision_point(image_bound, Point([point[2][0], point[2][1]])):
                                crop_img = window.copy()
                                x1_p = point[0][0]
                                y1_p = point[0][1]
                                x2_p = point[2][0]
                                y2_p = point[2][1]
                                point = [(x1_p - x, y1_p - y), (x2_p - x, y2_p - y)]
                                points.append(point)
                    else:
                        centd = Polygon(point).centroid
                        final_point = []
                        if self.check_collision_point(image_bound, centd):
                            crop_img = window.copy()
                            for pixel in point:
                                point1= [pixel[0]-x,pixel[1]-y]
                                final_point.append(point1)
                            points.append(final_point)

                if points:
                    filter_objects[key] = points
            sumo = 0
            for key in filter_objects.keys():
                sumo += len(filter_objects[key])
            if (sumo >= 1):
                if (flag == 0):
                    rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(self.outputDir + '/' + "image" + str(i) + '.jpg', rgb)
                    # imsave(self.outputDir + '/' + "image" + str(i) + '.jpg', crop_img[:,:,:3]) #using this will reduce cv2 import
                    flag = 1
                if not os.path.exists(self.outputDir + '/' + self.out_type):
                    os.mkdir(self.outputDir + '/' + self.out_type)
                if self.out_type =='xml':
                    self.create_xml_file(folder_name=self.out_type, file_name='image' + str(i), img_path_dir=self.outputDir,
                                         kml=filter_objects, xrect=xrect)
                else:
                    self.create_json_file(folder_name=self.out_type, file_name='image' + str(i), img_path_dir=self.outputDir,
                                     kml=filter_objects, xrect=xrect)
        print('File Writing Ended')

raster='/home/vasanth/Documents/data_collection/EPC/ukarine_data_collection/clipped_bound.tif'
vector_folders='/home/vasanth/Documents/data_collection/EPC/ukarine_data_collection/shp'
output ='/home/vasanth/Documents/data_collection/EPC/ukarine_data_collection/new'
boundary='/home/vasanth/Documents/data_collection/EPC/ukarine_data_collection/bound/clipped_bound.shp'

# new = data_generator(raster,vector_folders,output)
# new1 = data_generator(raster,vector_folders,output,boundary=boundary)


