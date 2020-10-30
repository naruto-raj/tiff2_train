import os
import time
import gdal
import base64
import json
from pyproj import Proj, transform
from skimage.io import imread, imsave
import lxml.etree as etree
import cv2
from osgeo import gdal, osr
import rasterio
import rasterio.mask
from shapely.geometry import Polygon, Point
import geopandas as gpd


class data_generator():

    def __init__(self, tifffile, vector_list, outputdir, window_w=1000, window_h=1000, step_size=500, boundary=None,
                 out_type='json'):
        self.input_tiff = tifffile
        self.input_vector = vector_list
        self.winW = int(window_w)
        self.winH = int(window_h)
        self.steps = int(step_size)
        self.bound = boundary
        self.outputDir = outputdir
        self.out_type = out_type
        self.objects = {}
        for vector in os.listdir(self.input_vector):
            self.objects[vector[:-4]] = []
        self.img = imread(self.input_tiff)
        image = gdal.Open(self.input_tiff)
        self.geoTrans = image.GetGeoTransform()
        print("Image Loaded Successfully")
        self.read_all_vectors()
        self.label_gen()

    def Pixel2world(self, geoMatrix, x, y):
        """Convert pixel coordinates to world coordinates W.R.T Geotiff"""
        return (geoMatrix[0] + (x * geoMatrix[1])), (geoMatrix[3] + (y * geoMatrix[5]))

    def world2Pixel(self, geoMatrix, x, y):
        """Convert world coordinates to pixel coordinates W.R.T Geotiff"""
        return (x - geoMatrix[0]) / geoMatrix[1], (y - geoMatrix[3]) / geoMatrix[5]

    def vector_reader(self, vector_file_path):
        """Function to read coordinates from shapefile"""
        print(vector_file_path)
        vector_read = gpd.read_file(vector_file_path, driver='ESRI Shapefile')
        new_polygon = []
        for i in vector_read.geometry:
            outer_coords = i.exterior.coords.xy
            final_coords = [(outer_coords[0][j], outer_coords[1][j]) for j in range(len(outer_coords[0]))]
            new_polygon.append([list(self.world2Pixel(self.geoTrans, coord[0], coord[1])) for coord in final_coords])
        return new_polygon

    def read_all_vectors(self):
        for file_name in os.listdir(self.input_vector):
            if file_name.endswith(".shp"):
                vector_file_path = os.path.join(self.input_vector, file_name)
                file_name = file_name.split('/')[-1][:-4]
                self.objects[file_name] = self.vector_reader(vector_file_path)

    @staticmethod
    def midpoint(p1, p2):
        """Calculate midpoint between two points"""
        return Point((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

    @staticmethod
    def check_collision_point(bound, detected):
        """Function to check if polygons are inside another polygon"""
        return Polygon(bound).contains(detected)

    @staticmethod
    def sliding_window(image, stepsize, windowsize):
        """slide a window across the image"""
        for y in range(0, image.shape[0], stepsize):
            for x in range(0, image.shape[1], stepsize):
                print(x, y, windowSize)
                yield x, y, image[y:y + windowsize[1], x:x + windowsize[0]]

    def crop_tiff(self):
        bound_read = gpd.read_file(self.bound, driver='ESRI Shapefile')
        feature = bound_read['geometry']
        with rasterio.open(self.input_tiff, "r") as src:
            out_image, out_transform = rasterio.mask.mask(src, feature, crop=True)
            out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                         "height": out_image.shape[1],
                         "width": out_image.shape[2],
                         "transform": out_transform})
        with rasterio.open(self.bound[:-4] + ".tif", "w", **out_meta) as dest:
            dest.write(out_image)
        return self.bound[:-4] + ".tif"

    def create_object(self, annotation, nam, kml, ind):
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

    def create_xml_file(self, folder_name, file_name, img_path_dir, kml, xrect):
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
        for key in kml.keys():
            if len(kml[key]) > 0:
                for j in range(0, len(kml[key])):
                    self.create_object(annotation=annotation, nam=key, kml=kml, ind=j)
        mydata = etree.tostring(annotation, pretty_print=True)
        myfile = open(img_path_dir + '/' + folder_n + '/' + filenameg + '.xml', 'wb')
        myfile.write(mydata)

    @staticmethod
    def create_json_file(folder_name, file_name, img_path_dir, kml, xrect):
        folder_n = folder_name
        filenameg = file_name
        path_n = img_path_dir + '/' + '/' + filenameg + '.jpg'
        with open(path_n, mode='rb') as file:
            img = file.read()
        imagedata = base64.b64encode(img).decode("utf-8")
        s_width = int(xrect[2] - xrect[0])
        s_height = int(xrect[3] - xrect[1])
        json_initial = {
            "version": "4.2.9",
            "flags": {},
            "shapes": [],
            "imagePath": path_n,
            "imageData": imagedata,
            "imageHeight": s_height,
            "imageWidth": s_width,
        }
        for key in kml.keys():
            if len(kml[key]) > 0:
                for j in range(0, len(kml[key])):
                    shape_data = {"label": key, "points": kml[key][j], "shape_type": "polygon", "flags": {}}
                    json_initial['shapes'].append(shape_data)
        myfile = open(img_path_dir + '/' + folder_n + '/' + filenameg + '.json', 'w')
        json.dump(json_initial, myfile)

    def label_gen(self):
        if self.bound is not None:
            result = self.crop_tiff()
            self.img = imread(result)
        for i,(x, y, window) in enumerate(self.sliding_window(self.img, stepSize=self.steps, windowSize=(self.winW, self.winH))):
            filter_objects = {}
            for label in os.listdir(self.input_vector):
                filter_objects[label[:-4]] = []
            image_bound = [(x, y), (x, y + self.winH), (x + self.winW, y + self.winH), (x + self.winW, y)]
            xrect = [x, y, x + self.winH, y + self.winW]
            for key in self.objects.keys():
                points = []
                for point in self.objects[key]:
                    if self.check_collision_point(image_bound, Point([point[0][0], point[0][1]])):
                        if self.check_collision_point(image_bound, Point([point[2][0], point[2][1]])):
                            crop_img = window.copy()
                            points.append([(point[0][0] - x, point[0][1] - y), (point[2][0] - x, point[2][1] - y)])
                if points:
                    filter_objects[key] = points
            sumo = 0
            for key in filter_objects.keys():
                sumo += len(filter_objects[key])
            if sumo >= 1:
                rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(self.outputDir + '/' + "image" + str(i) + '.jpg', rgb)
                if not os.path.exists(self.outputDir + '/' + self.out_type):
                    os.mkdir(self.outputDir + '/' + self.out_type)
                if self.out_type == 'xml':
                    self.create_xml_file(folder_name=self.out_type, file_name='image' + str(i),
                                         img_path_dir=self.outputDir,
                                         kml=filter_objects, xrect=xrect)
                else:
                    self.create_json_file(folder_name=self.out_type, file_name='image' + str(i),
                                          img_path_dir=self.outputDir,
                                          kml=filter_objects, xrect=xrect)
        print('File Writing Ended')