import argparse
import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
import duckdb
from rasterio.mask import mask
from shapely.geometry import Polygon
from shapely import wkt
from pyproj import Transformer
from exactextract import exact_extract
import sqlite3
import os
import sys
import time

def overture_query(xmin, xmax, ymin, ymax, overture_version):
    """
    Queries Overture data for building footprints within a specified bounding box.
    
    Parameters:
    - xmin, xmax, ymin, ymax: Coordinates defining the bounding box for the query
    - overture_version: The version of Overture data to query (e.g., '2024-07-22.0')
    
    Returns:
    - GeoDataFrame containing the queried building footprints from Overture
    """
    con = duckdb.connect(database=':memory:')  # Using an in-memory database
    
    start_time = time.time()
    
    # Construct the SQL query to read the Overture data for the specified bounding box
    query = f"""
        LOAD spatial;
        LOAD httpfs;
        SET enable_geoparquet_conversion = false;  -- Disable geoparquet conversion
        SELECT
            id,
            ST_AsText(ST_GeomFromWKB(geometry)) as geometry
        FROM read_parquet('s3://overturemaps-us-west-2/release/{overture_version}/theme=buildings/type=*/*')
        WHERE
            bbox.xmin >= {xmin}
            AND bbox.xmax <= {xmax}
            AND bbox.ymin >= {ymin}
            AND bbox.ymax <= {ymax}
    """
    
    # Execute the query and fetch the result into a Pandas DataFrame
    result_df = con.execute(query).fetchdf()
    result_df['geometry'] = result_df['geometry'].apply(wkt.loads)
    
    # Convert the result to a GeoDataFrame (GeoPandas requires a Geometry column)
    gdf = gpd.GeoDataFrame(result_df, geometry='geometry', crs="EPSG:4326")
    
    # Calculate and print execution time
    end_time = time.time()
    if verbose:
        print(f"Overture query execution time: {end_time - start_time:.2f} seconds")
    
    con.close()
    
    return gdf

def create_bbox_geom(bounds, grid_size_x, grid_size_y):
    """Enlarges and aligns the bounding box to the grid cell size."""
    
    # Align the left and right sides to the grid size in the X dimension
    left = np.round(bounds[0] / grid_size_x) * grid_size_x
    right = np.round(bounds[2] / grid_size_x) * grid_size_x
    
    # Align the bottom and top sides to the grid size in the Y dimension
    bottom = np.round(bounds[1] / grid_size_y) * grid_size_y
    top = np.round(bounds[3] / grid_size_y) * grid_size_y
    
    return Polygon([(left, bottom), (left, top), (right, top), (right, bottom), (left, bottom)])

def clip_raster(raster_path, bbox_geom, output_path, in_memory=False):
    """Clip the raster based on the provided geometry."""
    start_time = time.time()
    try:
        with rasterio.open(raster_path) as src:
            out_image, out_transform = mask(src, [bbox_geom], crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": src.nodata
            })
            if in_memory:
                if verbose:
                    print(f"Raster clipping (in memory) time: {time.time() - start_time:.2f} seconds")
                return out_image, out_meta  # Return clipped data in memory
            else:
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)
                if verbose:
                    print(f"Raster clipping (to file) time: {time.time() - start_time:.2f} seconds")
                return None
    except Exception as e:
        print(f"Error processing raster file: {e}")
        sys.exit(1)

def compute_centroid_coordinates(gdf):
    """Compute the centroid coordinates (latitude, longitude) of each polygon."""
    start_time = time.time()

    # Check if the current CRS is not EPSG:4326 (WGS84)
    if gdf.crs != 'EPSG:4326':
        # Define the transformer to convert coordinates from the current CRS to WGS84 (EPSG:4326)
        transformer = Transformer.from_crs(gdf.crs, 'EPSG:4326', always_xy=True)
    else:
        # If the CRS is already EPSG:4326, no transformation is needed
        transformer = None
    
    # Compute the centroid of each geometry in the GeoDataFrame
    gdf['centroid'] = gdf.geometry.centroid
    
    # If the transformation is needed, apply it to each centroid
    if transformer:
        gdf['lon'], gdf['lat'] = zip(*gdf['centroid'].apply(lambda geom: transformer.transform(geom.x, geom.y)))
    else:
        # If already in EPSG:4326, just extract the x, y directly
        gdf['lon'] = gdf['centroid'].x.round(5)
        gdf['lat'] = gdf['centroid'].y.round(5)
    
    gdf.drop(columns=['centroid'], inplace=True)
    if verbose:
        print(f"Centroid computation time: {time.time() - start_time:.2f} seconds")
    return gdf

def calculate_metrics_in_utm(gdf):
    """Calculate metrics (area, perimeter, shapefactor) in UTM projection."""
    start_time = time.time()
    
    def get_utm_crs(lon, lat):
        """Helper function to get UTM CRS based on coordinates."""
        zone_number = int((lon + 180) / 6) + 1
        is_northern = lat >= 0
        return f"EPSG:326{zone_number}" if is_northern else f"EPSG:327{zone_number}"
    
    # Ensure the GeoDataFrame is in WGS84 (EPSG:4326)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    
    # Get UTM CRS for each row based on coordinates
    gdf['utm_crs'] = gdf.apply(lambda row: get_utm_crs(row['lon'], row['lat']), axis=1)
    
    # Group by UTM CRS to handle each UTM zone separately
    grouped = gdf.groupby('utm_crs')
    metrics = []
    
    for utm_crs, group in grouped:
        if verbose:
            print(f"Processing UTM CRS: {utm_crs}")
        group_utm = group.to_crs(utm_crs)
        area = group_utm.geometry.area
        perimeter = group_utm.geometry.length
        shapefactor = (2 / group_utm['height'] + perimeter / area).round(5)
        
        group_metrics = pd.DataFrame({
            'area': area.round(2),
            'perimeter': perimeter.round(2),
            'shapefactor': shapefactor,
        }, index=group.index)
        
        metrics.append(group_metrics)
    
    metrics_df = pd.concat(metrics)
    if verbose:
        print(f"UTM metric calculation time: {time.time() - start_time:.2f} seconds")
    
    return metrics_df[['area', 'perimeter', 'shapefactor']]

def calculate_zonal_stats(gdf, raster_paths, outpath):
    """ Calculates zonal statistics for each polygon in the GeoDataFrame using specified raster layers. """
    for field_name, raster_path in raster_paths.items():
        if verbose:
            print(f"Processing raster: {field_name}")
        start_time = time.time()
        
        # Open the raster to get its CRS
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs  # Get the CRS of the raster
            grid_size_x, grid_size_y = src.res  # Get the pixel size in both x and y dimensions
            if verbose:
                print(f"Raster CRS: {raster_crs}")
                print(f"Raster Resolution (Grid Size): {grid_size_x}, {grid_size_y}")
        
        # Check if the CRS of the gdf and the raster are the same
        if gdf.crs != raster_crs:
            if verbose:
                print(f"CRS mismatch: Reprojecting GeoDataFrame from {gdf.crs} to {raster_crs}")
            gdf = gdf.to_crs(raster_crs)  # Reproject gdf to the raster's CRS
        
        # Align the bounds of the GeoDataFrame to the input grid
        bbox_geom = create_bbox_geom(gdf.total_bounds, grid_size_x, grid_size_y)
        
        # Clip raster to the bounding box geometry
        clipped_raster_path = os.path.join(outpath, f"{field_name}_clipped.tif")
        
        # Assuming bbox_geom is the bounding geometry for the raster to clip, which is derived from the gdf or specific AOI
        clip_raster(raster_path, bbox_geom, clipped_raster_path, in_memory=False)
        
        try:
            if field_name == "height_1" or field_name == "height_2":
                # For continuous data (height), calculate the mean value
                stats_df = exact_extract(clipped_raster_path, gdf, 'mean', output='pandas')
                gdf[field_name] = stats_df['mean'].round(2)
            else:
                # For categorical data (use, age), calculate the majority value
                stats_df = exact_extract(clipped_raster_path, gdf, 'majority', output='pandas')
                gdf[field_name] = stats_df['majority'].astype('int8')
        except Exception as e:
            print(f"Error processing raster {field_name}: {e}")
            continue
        
        if verbose:
            print(f"Zonal statistics computed for {field_name}, took {time.time() - start_time:.2f} seconds")
        
        # Optionally remove the temporary clipped raster after processing
        try:
            os.remove(clipped_raster_path)
        except Exception as e:
            print(f"Error deleting temporary raster: {e}")
    
    return gdf

def apply_vacuum_to_gpkg(gpkg_path):
    """Applies VACUUM to the GeoPackage to optimize the database by removing unused space."""
    try:
        conn = sqlite3.connect(gpkg_path)
        cursor = conn.cursor()
        cursor.execute("VACUUM")
        conn.commit()
        conn.close()
        if verbose:
            print(f"VACUUM applied successfully to {gpkg_path}")
    except sqlite3.Error as e:
        print(f"Error applying VACUUM to GeoPackage: {e}")
        sys.exit(1)

def process_and_integrate_building_attributes(input_file, outpath, overture_version, min_lat, max_lat, min_lon, max_lon, 
                                              use_raster_1=None, use_raster_2=None, age_raster=None, 
                                              height_raster_1=None, height_raster_2=None):
    """Processes building data by integrating raster attributes and other relevant information into a GeoPackage."""
    
    if outpath is None:
        print("Error: Output path (outpath) must be specified.")
        sys.exit(1)

    start_proc_time = time.time()
    
    if overture_version:
        input_name = f"overture_buildings_{min_lon}_{max_lon}_{min_lat}_{max_lat}_{overture_version}"
    else:
        input_name = os.path.splitext(os.path.basename(input_file))[0]
    
    # Print the initial processing message
    print(f"Processing {input_name} ...")
    
    if overture_version:
        xmin, ymin, xmax, ymax = min_lon, min_lat, max_lon, max_lat
        gdf = overture_query(xmin, xmax, ymin, ymax, overture_version)
        if verbose:
            print(f"Overture data downloaded directly into memory")
        output_filename = os.path.join(outpath, f"overture_buildings_{xmin}_{xmax}_{ymin}_{ymax}_{overture_version}_integration.gpkg")
    else:
        gdf = gpd.read_file(input_file)
        output_filename = os.path.join(outpath, f"{os.path.splitext(os.path.basename(input_file))[0]}_integration.gpkg")
        if verbose:
            print("Data loaded from input GeoPackage.")
    
    # Get input fields
    input_fields = [col for col in gdf.columns]
    
    # Original CRS before reprojecting
    original_crs = gdf.crs
    
    # Build the raster_paths dictionary based on available rasters
    raster_paths = {}
    
    # Include use rasters if provided
    if use_raster_1:
        raster_paths["use_1"] = use_raster_1
    if use_raster_2:
        raster_paths["use_2"] = use_raster_2
    
    # Include age raster if provided
    if age_raster:
        raster_paths["epoch"] = age_raster
    
    # Include height rasters if provided
    if height_raster_2:
        raster_paths["height_2"] = height_raster_2
    if height_raster_1:
        raster_paths["height_1"] = height_raster_1

    # Check that there are raster layers available
    if not raster_paths:
        print("Error: No raster layers provided. At least one raster layer is required.")
        sys.exit(1)
    
    # Perform zonal statistics using the raster_paths dictionary
    gdf = calculate_zonal_stats(gdf, raster_paths, outpath)
    
    # Merge the two height fields into one
    start_time = time.time()
    if 'height_1' not in gdf.columns and 'height_2' not in gdf.columns:
        gdf['height'] = np.nan  # If both are missing, set 'height' to NaN
    elif 'height_1' not in gdf.columns:
        gdf['height'] = gdf['height_2']  # If height_1 is missing, height height_2
    elif 'height_2' not in gdf.columns:
        gdf['height'] = gdf['height_1']  # If height_2 is missing, height height_1
    else:
        gdf['height'] = np.where(gdf['height_1'] == 0, gdf['height_2'], gdf['height_1'])  # Merge height_1 and height_2
    if verbose:
        print(f"Merged height fields, took {time.time() - start_time:.2f} seconds")
    # Drop intermediate columns if exist
    for col in ['height_1', 'height_2']:
        if col in gdf.columns:
            gdf.drop(columns=[col], inplace=True)
    
    # Merge the two use fields into one
    start_time = time.time()
    if 'use_1' not in gdf.columns and 'use_2' not in gdf.columns:
        gdf['use'] = np.nan  # If both are missing, set 'use' to NaN
    elif 'use_1' not in gdf.columns:
        gdf['use'] = gdf['use_2']  # If use_1 is missing, use use_2
    elif 'use_2' not in gdf.columns:
        gdf['use'] = gdf['use_1']  # If use_2 is missing, use use_1
    else:
        gdf['use'] = np.where(gdf['use_1'] == 0, gdf['use_2'], gdf['use_1'])  # Merge use_1 and use_2
    if verbose:
        print(f"Merged use fields, took {time.time() - start_time:.2f} seconds")
    # Drop intermediate columns if exist
    for col in ['use_1', 'use_2']:
        if col in gdf.columns:
            gdf.drop(columns=[col], inplace=True)
    
    # Compute centroid coordinates (in WGS84)
    gdf = compute_centroid_coordinates(gdf)
    
    # Calculate area, perimeter, and shape factor in UTM and join with original GDF
    utm_metrics = calculate_metrics_in_utm(gdf)
    gdf = gdf.join(utm_metrics)
    
    # Add the height, shapefactor, area, perimeter, etc., to the list of columns
    final_fields = input_fields + ['height', 'shapefactor', 'use', 'epoch', 'area', 'perimeter']
    
    # Reorder fields according to the specified order
    gdf = gdf[final_fields]
    
    # Reproject back to the original CRS
    if original_crs != "ESRI:54009":
        gdf = gdf.to_crs(original_crs)
        if verbose:
            print(f"Reprojected back to the original CRS: {original_crs}.")
    
    gdf.to_file(output_filename, driver='GPKG')
    if verbose:
        print(f"Created {output_filename} with integrated attributes.")
    
    apply_vacuum_to_gpkg(output_filename)
    
    end_proc_time = time.time()
    processing_time = end_proc_time - start_proc_time
    print(f"Processing completed in {processing_time:.2f} seconds")

if __name__ == "__main__":
    # ArgumentParser setup with description and examples
    parser = argparse.ArgumentParser(
        description="Assign attributes to a GeoPackage or process Overture data for buildings and associated attributes.",
        epilog="Examples:\n\n1. Process a GeoPackage file:\n   python3 GHSL_building_data_integrator.py /path/to/input.gpkg --use_raster_1 /path/to/use_1.tif --use_raster_2 /path/to/use_2.tif --age_raster /path/to/age.tif --height_raster_1 /path/to/height_1.tif --height_raster_2 /path/to/height_2.tif\n\n2. Process Overture data with a bounding box and a specific version:\n   python3 GHSL_building_data_integrator.py --overture --overture_version 2024-07-22.0 --min_lat 10.0 --max_lat 20.0 --min_lon -80.0 --max_lon -70.0 --use_raster_1 /path/to/use_1.tif --use_raster_2 /path/to/use_2.tif --age_raster /path/to/age.tif --height_raster_1 /path/to/height_1.tif --height_raster_2 /path/to/height_2.tif",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Define the arguments
    parser.add_argument("--overture", action="store_true", help="Flag to use Overture data")
    parser.add_argument("--verbose", action="store_true", help="Flag to enable verbose output")
    parser.add_argument("input_file", type=str, help="Path to the input GeoPackage or shapefile", nargs='?')
    parser.add_argument("--outpath", type=str, required=True, help="Output directory path")
    
    # Overture arguments: required only if `--overture` is passed
    parser.add_argument("--overture_version", type=str, help="Overture version (e.g., 2024-07-22.0)", required=False)
    parser.add_argument("--min_lat", type=float, help="Minimum latitude", required=False)
    parser.add_argument("--max_lat", type=float, help="Maximum latitude", required=False)
    parser.add_argument("--min_lon", type=float, help="Minimum longitude", required=False)
    parser.add_argument("--max_lon", type=float, help="Maximum longitude", required=False)
    
    # Optional raster input paths
    parser.add_argument("--use_raster_1", type=str, help="Path to the first use raster", required=False)
    parser.add_argument("--use_raster_2", type=str, help="Path to the second use raster", required=False)
    parser.add_argument("--age_raster", type=str, help="Path to the age raster", required=False)
    parser.add_argument("--height_raster_1", type=str, help="Path to the first height raster", required=False)
    parser.add_argument("--height_raster_2", type=str, help="Path to the second height raster", required=False)

    # Parsing the arguments
    args = parser.parse_args()
    verbose = args.verbose
    
    # Conditional argument handling based on the presence of `--overture`
    if args.overture:
        if not (args.overture_version and args.min_lat and args.max_lat and args.min_lon and args.max_lon):
            parser.print_help()
            print("Error: --overture_version, --min_lat, --max_lat, --min_lon, and --max_lon must be specified for Overture data.")
            sys.exit(1)
        process_and_integrate_building_attributes(None, args.overture_version, args.min_lat, args.max_lat, args.min_lon, args.max_lon,
                                   args.use_raster_1, args.use_raster_2, args.age_raster, args.height_raster_1, args.height_raster_2)
    else:
        if not args.input_file:
            parser.print_help()
            print("Error: Input GeoPackage must be specified for non-Overture processing.")
            sys.exit(1)
        process_and_integrate_building_attributes(args.input_file, args.outpath, None, None, None, None, None,
                                                  args.use_raster_1, args.use_raster_2, args.age_raster, args.height_raster_1, args.height_raster_2)
