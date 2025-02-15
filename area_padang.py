from datetime import datetime
import json
import os
import math
import logging
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Tuple, List, Dict, Optional
from roboflow import Roboflow
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point, Polygon, shape
from shapely.ops import transform
import pyproj
from functools import partial
import geojson

def load_geojson_polygon(filename: str) -> Polygon:
    """Load and parse GeoJSON polygon."""
    with open(filename, 'r') as f:
        geojson_data = geojson.load(f)
    
    # Get the first polygon from features
    feature = geojson_data['features'][0]
    return shape(feature['geometry'])

def point_in_polygon(lat: float, lon: float, polygon: Polygon) -> bool:
    """Check if a point is inside the polygon."""
    point = Point(lon, lat)  # Note: GeoJSON uses (lon, lat) order
    return polygon.contains(point)

def calculate_bounds(polygon: Polygon) -> dict:
    """Calculate the bounding box of the polygon."""
    bounds = polygon.bounds  # (minx, miny, maxx, maxy)
    return {
        'west': bounds[0],
        'south': bounds[1],
        'east': bounds[2],
        'north': bounds[3]
    }

def filter_buildings_in_polygon(buildings: List[dict], polygon: Polygon) -> List[dict]:
    """Filter buildings that fall within the polygon."""
    buildings_in_polygon = []
    seen_boxes = set()
    
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union of two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (area1 + area2 - intersection)
    
    for building in buildings:
        # Check if building center is within polygon
        if point_in_polygon(building['latitude'], building['longitude'], polygon):
            box = building['box']
            
            # Check for duplicates using IOU
            is_duplicate = False
            for seen_box in seen_boxes:
                if calculate_iou(box, seen_box) > 0.5:  # IOU threshold
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                buildings_in_polygon.append(building)
                seen_boxes.add(box)
    
    return buildings_in_polygon

def process_location_with_polygon(geojson_path: str) -> Dict:
    try:
        # Load polygon from GeoJSON
        polygon = load_geojson_polygon(geojson_path)
        bounds = calculate_bounds(polygon)
        
        logger.info(f"Processing area within polygon bounds: N:{bounds['north']}, S:{bounds['south']}, E:{bounds['east']}, W:{bounds['west']}")
        
        # Load YOLO model
        model = load_or_train_model('building_detector.pt')
        
        # Calculate center point of polygon for initial reference
        centroid = polygon.centroid
        center_lat, center_lng = centroid.y, centroid.x
        
        # Calculate maximum distance from center to edge of polygon
        # to determine coverage area
        geod = pyproj.Geod(ellps='WGS84')
        max_distance = 0
        for coord in polygon.exterior.coords:
            _, _, distance = geod.inv(center_lng, center_lat, coord[0], coord[1])
            max_distance = max(max_distance, distance)
        
        zoom = 18  # Maximum zoom for detail
        tile_radius = calculate_tile_radius(center_lat, center_lng, max_distance, zoom)
        logger.info(f"Area requires {tile_radius} tiles radius for coverage")
        
        # Process area and get all buildings
        image, buildings = process_large_area(
            center_lat, center_lng, tile_radius,
            max_distance, model, zoom=zoom
        )
        
        # Filter buildings to only those within polygon
        buildings_in_polygon = filter_buildings_in_polygon(buildings, polygon)
        
        logger.info(f"Found {len(buildings_in_polygon)} buildings within polygon area")
        
        return save_building_coordinates(buildings_in_polygon)
    
    except Exception as e:
        logger.error(f"Error processing location: {str(e)}")
        raise

# Main execution


# Constants
TILE_SIZE = 256
EARTH_RADIUS = 6371000
MAX_CONFIDENCE = 1
MAX_OVERLAP = 50
USER_AGENT = 'BuildingDetectionBot/1.0'

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Utility Functions
def visualize_detections(image: np.ndarray, buildings: List[dict], 
                        buildings_in_radius: List[dict],
                        center_x: float, center_y: float, 
                        radius_pixels: float):
    """Visualize building detections with bounding boxes."""
    
    # Create figure and axes
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    # Draw all detected buildings in gray
    for building in buildings:
        box = building['box']
        rect = patches.Rectangle(
            (box[0], box[1]), 
            box[2] - box[0], 
            box[3] - box[1],
            linewidth=1, 
            edgecolor='gray', 
            facecolor='none'
        )
        ax.add_patch(rect)
    
    # Draw buildings within radius in red
    for building in buildings_in_radius:
        box = building['box']
        rect = patches.Rectangle(
            (box[0], box[1]), 
            box[2] - box[0], 
            box[3] - box[1],
            linewidth=2, 
            edgecolor='red', 
            facecolor='none'
        )
        ax.add_patch(rect)
    
    # Draw search radius circle
    circle = plt.Circle(
        (center_x, center_y), 
        radius_pixels, 
        color='blue', 
        fill=False, 
        linestyle='--'
    )
    ax.add_patch(circle)
    
    # Add legend
    ax.plot([], [], color='gray', label='All Detections')
    ax.plot([], [], color='red', label='In Radius')
    ax.plot([], [], color='blue', linestyle='--', label='Search Radius')
    plt.legend()
    
    plt.axis('off')
    plt.show()

def load_or_train_model(weights_path: str = 'building_detector.pt') -> YOLO:
    """Load existing model from path or train a new one."""
    try:
        if os.path.exists(weights_path):
            logger.info(f"Loading existing model from {weights_path}")
            return YOLO(weights_path)
        else:
            raise FileNotFoundError(f"Model not found at {weights_path}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def calculate_box_center(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """Calculate center point of bounding box in pixels."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def calculate_tile_radius(latitude: float, longitude: float, search_radius_meters: float, zoom: int) -> int:
    """Calculate minimum required tile radius based on search radius."""
    lat_rad = math.radians(latitude)
    meters_per_pixel = (156543.03392 * math.cos(lat_rad)) / (2 ** zoom)
    meters_per_tile = meters_per_pixel * TILE_SIZE
    
    # Add safety margin for tile coverage
    tiles_needed = math.ceil((search_radius_meters * 2) / meters_per_tile)  # Multiply by 2 for diameter
    
    return max(1, tiles_needed)

def tile_to_latlng(x: float, y: float, zoom: int) -> Tuple[float, float]:
    """Convert tile coordinates to latitude/longitude."""
    n = 2.0 ** zoom
    lon_deg = x / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates in meters."""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) ** 2)
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS * c

def is_point_in_circle(point_x: float, point_y: float, circle_x: float, circle_y: float, radius: float) -> bool:
    """Check if point is within circle using pixel coordinates."""
    dx = point_x - circle_x
    dy = point_y - circle_y
    return (dx * dx + dy * dy) <= (radius * radius)

def get_surrounding_tiles(center_lat: float, center_lng: float, zoom: int, tile_radius: int) -> List[Tuple[int, int]]:
    """Get coordinates of surrounding tiles."""
    lat_rad = math.radians(center_lat)
    n = 2.0 ** zoom
    center_x = int((center_lng + 180.0) / 360.0 * n)
    center_y = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    
    return [(x, y) for x in range(center_x - tile_radius, center_x + tile_radius + 1)
            for y in range(center_y - tile_radius, center_y + tile_radius + 1)]

def calculate_map_bounds(center_lat: float, center_lng: float, zoom: int, image_size: Tuple[int, int]) -> Dict:
    """Calculate geographical bounds of a map image."""
    WORLD_SIZE = TILE_SIZE * (2 ** zoom)
    lat_rad = math.radians(center_lat)
    ground_resolution = (math.cos(lat_rad) * 2 * math.pi * EARTH_RADIUS) / WORLD_SIZE

    width_meters = image_size[0] * ground_resolution
    height_meters = image_size[1] * ground_resolution
    lat_diff = (height_meters / 2) / 111320
    lng_diff = (width_meters / 2) / (111320 * math.cos(lat_rad))

    return {
        'north': center_lat + lat_diff,
        'south': center_lat - lat_diff,
        'east': center_lng + lng_diff,
        'west': center_lng - lng_diff
    }

# def save_building_coordinates(buildings: List[dict]) -> str:
#     """Save building coordinates in JSON format."""

#     formatted_data = {
#         "data": [
#             {
#                 "id": i + 1,
#                 "longitude": float(building['longitude']),
#                 "latitude": float(building['latitude'])
#             }
#             for i, building in enumerate(buildings)
#         ]
#     }

#     return formatted_data



def process_large_area(latitude: float, longitude: float, tile_radius: int,
                      search_radius: float, model, zoom: int = 18) -> Tuple[np.ndarray, List[dict]]:
    """Process a large area for building detection with adjusted radius."""
    # Get surrounding tiles
    tiles = get_surrounding_tiles(latitude, longitude, zoom, tile_radius)
    min_x = min(t[0] for t in tiles)
    min_y = min(t[1] for t in tiles)

    # Calculate combined image dimensions
    width = (max(t[0] for t in tiles) - min_x + 1) * TILE_SIZE
    height = (max(t[1] for t in tiles) - min_y + 1) * TILE_SIZE
    combined_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Adjust search radius to fit within image bounds
    adjusted_radius = adjust_search_radius(search_radius, (width, height), latitude, zoom)
    if adjusted_radius != search_radius:
        print(f"Adjusted search radius from {search_radius}m to {adjusted_radius}m to fit display")

    all_buildings = []

    # Process each tile
    for x, y in tiles:
        try:
            tile_offset = (x - min_x, y - min_y)
            tile_array, tile_buildings = process_single_tile(
                x, y, zoom, model, tile_offset, latitude, longitude
            )

            # Place tile in combined image
            pos_x = tile_offset[0] * TILE_SIZE
            pos_y = tile_offset[1] * TILE_SIZE
            combined_image[pos_y:pos_y + TILE_SIZE, pos_x:pos_x + TILE_SIZE] = tile_array

            all_buildings.extend(tile_buildings)

        except Exception as e:
            print(f"Error processing tile {x},{y}: {e}")

    # Filter buildings within adjusted radius
    buildings_in_radius = filter_buildings_in_radius(
        all_buildings, 
        latitude, 
        longitude, 
        adjusted_radius,
        width, 
        height, 
        zoom
    )

    # Visualize results
    center_x = width / 2
    center_y = height / 2
    lat_rad = math.radians(latitude)
    meters_per_pixel = (156543.03392 * math.cos(lat_rad)) / (2 ** zoom)
    radius_pixels = (adjusted_radius / meters_per_pixel)
    
    visualize_detections(
        combined_image, 
        all_buildings,
        buildings_in_radius,
        center_x,
        center_y,
        radius_pixels
    )
    
    logger.info(f"Completed processing. Found {len(buildings_in_radius)} buildings within {adjusted_radius}m radius")
    return combined_image, buildings_in_radius

    return combined_image, buildings_in_radius

def filter_buildings_in_radius(buildings: List[dict], center_lat: float, 
                             center_lng: float, radius_meters: float,
                             image_width: int, image_height: int,
                             zoom: int) -> List[dict]:
    """Filter buildings within a specified radius with duplicate handling."""
    buildings_in_radius = []
    seen_boxes = set()
    
    # Calculate circle center in pixels
    center_x = image_width / 2
    center_y = image_height / 2
    
    # Convert radius to pixels with safety margin
    lat_rad = math.radians(center_lat)
    meters_per_pixel = (156543.03392 * math.cos(lat_rad)) / (2 ** zoom)
    radius_pixels = (radius_meters / meters_per_pixel) * 1.1  # Add 10% margin
    
    logger.debug(f"Processing {len(buildings)} raw detections")
    
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union of two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (area1 + area2 - intersection)
    
    for building in buildings:
        box = building['box']
        box_center_x = (box[0] + box[2]) / 2
        box_center_y = (box[1] + box[3]) / 2
        
        # Check if building is within radius
        if is_point_in_circle(box_center_x, box_center_y, center_x, center_y, radius_pixels):
            # Check for duplicates using IOU
            is_duplicate = False
            for seen_box in seen_boxes:
                if calculate_iou(box, seen_box) > 0.5:  # IOU threshold
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                buildings_in_radius.append(building)
                seen_boxes.add(box)
                
    logger.debug(f"Found {len(buildings_in_radius)} unique buildings after filtering")
    return buildings_in_radius

def process_single_tile(x_tile: int, y_tile: int, zoom: int, model: YOLO,
                       tile_offset: Tuple[int, int],
                       center_lat: float, center_lng: float) -> Tuple[np.ndarray, List[dict]]:
    """Process single tile with YOLO model."""
    url = f"https://tile.openstreetmap.org/{zoom}/{x_tile}/{y_tile}.png"
    response = requests.get(url, headers={'User-Agent': USER_AGENT})
    tile_img = Image.open(BytesIO(response.content)).convert('RGB')
    tile_array = np.array(tile_img)
    
    # Get predictions from YOLO model
    results = model.predict(tile_array, conf=0.5, iou=0.5)[0]
    buildings = []
    
    # Get tile corner coordinates
    tile_lat, tile_lng = tile_to_latlng(x_tile, y_tile, zoom)
    next_tile_lat, next_tile_lng = tile_to_latlng(x_tile + 1, y_tile + 1, zoom)
    
    for box in results.boxes:
        box_coords = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf.cpu().numpy())
        x1, y1, x2, y2 = box_coords
        
        # Calculate center point
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Convert to tile coordinates
        px_ratio = center_x / TILE_SIZE
        py_ratio = center_y / TILE_SIZE
        
        # Calculate lat/lng
        lat = tile_lat + (1 - py_ratio) * (next_tile_lat - tile_lat)
        lng = tile_lng + px_ratio * (next_tile_lng - tile_lng)
        
        # Global coordinates
        x_global = x1 + (tile_offset[0] * TILE_SIZE)
        y_global = y1 + (tile_offset[1] * TILE_SIZE)
        
        buildings.append({
            'confidence': confidence,
            'latitude': float(lat),
            'longitude': float(lng),
            'box': (
                x_global,
                y_global,
                x_global + (x2 - x1),
                y_global + (y2 - y1)
            )
        })
    
    return tile_array, buildings

def save_building_coordinates(buildings: List[dict]) -> Dict:
    """Save building coordinates in JSON format."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"building_coordinates_{timestamp}.json"
    
    formatted_data = {
        "data": [
            {
                "id": i + 1,
                "longitude": float(building['longitude']),
                "latitude": float(building['latitude'])
            }
            for i, building in enumerate(buildings)
        ]
    }

    try:
        with open(filename, 'w') as f:
            json.dump(formatted_data, f, indent=4)
        logger.info(f"Saved coordinates to {filename}")
    except IOError as e:
        logger.error(f"Failed to save coordinates: {str(e)}")
        
    return formatted_data

def adjust_search_radius(radius_meters: float, image_size: Tuple[int, int], 
                        center_lat: float, zoom: int) -> float:
    """Adjust search radius to fit within image bounds."""
    lat_rad = math.radians(center_lat)
    meters_per_pixel = (156543.03392 * math.cos(lat_rad)) / (2 ** zoom)
    
    # Calculate maximum radius that fits in image
    max_radius_pixels = min(image_size[0], image_size[1]) / 2
    max_radius_meters = max_radius_pixels * meters_per_pixel
    
    return min(radius_meters, max_radius_meters)

def process_location(latitude: float, longitude: float, search_radius: float) -> Dict:
    try:
        logger.info(f"Processing location: ({latitude}, {longitude}) with radius {search_radius}m")
        
        # Load YOLO model
        model = load_or_train_model('building_detector.pt')
        
        zoom = 18
        tile_radius = calculate_tile_radius(latitude, longitude, search_radius, zoom)
        logger.info(f"Search radius: {search_radius}m requires {tile_radius} tiles radius")
        
        # Calculate actual coverage
        lat_rad = math.radians(latitude)
        meters_per_pixel = (156543.03392 * math.cos(lat_rad)) / (2 ** zoom)
        meters_covered = tile_radius * TILE_SIZE * meters_per_pixel
        logger.info(f"Actual area covered: {meters_covered}m radius")
        
        image, buildings = process_large_area(
            latitude, longitude, tile_radius,
            search_radius, model, zoom=zoom
        )
        
        return save_building_coordinates(buildings)
    
    except Exception as e:
        logger.error(f"Error processing location: {str(e)}")
        raise

# Main execution
if __name__ == "__main__":
    GEOJSON_PATH = "padang.geojson"
    process_location_with_polygon(GEOJSON_PATH)
# 
#