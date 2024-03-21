import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point

def get_iris(areas, longitude, latitude):

    location = Point(longitude, latitude)
    polygon = areas.contains(location)
    
    code_iris = str(areas.loc[polygon, 'CODE_IRIS'].values[0])
    commune_iris = areas.loc[polygon, 'NOM_COM'].values[0]
    nom_iris = areas.loc[polygon, 'NOM_IRIS'].values[0]

    return code_iris, commune_iris, nom_iris

def get_iris_oneaddress(longitude, latitude):
    # Load the shapefile
    areas = gpd.read_file('assets/contours-iris/CONTOURS-IRIS.shp')
    areas = areas.to_crs('4326')
    areas['dep'] = areas['INSEE_COM'].apply(lambda x: x[:2])
    code_iris, commune_iris, nom_iris = get_iris(areas, longitude, latitude)
    return code_iris

