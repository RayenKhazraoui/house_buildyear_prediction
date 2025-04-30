import json
import requests
import math

# Load the JSON file
with open("key.json", "r") as file:
    config = json.load(file)

# Access the API key
API_KEY = config.get("google_api_key")

def calculate_heading_and_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the angle (bearing) and distance between two sets of coordinates.
    Returns the heading (in degrees, 0° to 360°) and distance (in kilometers).
    """
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Calculate the difference in longitude and latitude
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    # Calculate the x and y components for heading
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    
    # Calculate the initial bearing
    heading = math.atan2(x, y)
    
    # Convert radians to degrees and normalize to 0–360°
    heading = (math.degrees(heading) + 360) % 360

    # Haversine formula for distance
    R = 6371.0  # Earth's radius in kilometers
    a = math.sin(delta_lat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c *1000 # Distance in kilometers

    return heading, distance

def get_image(latitude, longitude, size, heading, pitch, fov, index):
    # Build the URL
    url = f"https://maps.googleapis.com/maps/api/streetview?size={size}&location={latitude},{longitude}&heading={heading}&pitch={pitch}&fov={fov}&key={API_KEY}"

    # Send the request to Google
    response = requests.get(url)

    image_path = rf"C:\Users\rkhaz\Documents\Drive\ml_fun\images_houses\{index}.jpg"

    # Check the response
    if response.status_code == 200:
        # Save the image
        with open(image_path, "wb") as file:
            file.write(response.content)
        print(f"Street View image saved as {index}.jpg'")

        return image_path

    else:
        print(f"Error: {response.status_code}, {response.text}")

def get_closest_streetview_coordinates_and_heading(target_lat, target_lon, api_key):
    # Street View Metadata API URL
    url = f"https://maps.googleapis.com/maps/api/streetview/metadata?location={target_lat},{target_lon}&key={api_key}"
    
    # Send the request
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        metadata = response.json()
        if metadata.get("status") == "OK":
            # Extract closest coordinates and original heading
            closest_location = metadata.get("location", {})
            pano_id = metadata.get("pano_id", None)
            heading = metadata.get("heading", 0)  # Default to 0 if not available
            
            # Return results
            return {
                "status": "success",
                "closest_coordinates": closest_location,
                "pano_id": pano_id,
                "original_heading": heading
            }
        else:
            # Return error with detailed status
            return {"status": "error", "message": metadata.get("status", "Unknown Error")}
    else:
        # Handle network errors
        return {"status": "error", "message": f"HTTP Error {response.status_code}: {response.text}"}

def retrieve_image(target_coordinates, index, max_distance):
    size = "640x640"
    pitch = 0  # Camera tilt (0 = straight ahead)
    fov = 90   # Field of view

    # Call the function to get metadata
    result = get_closest_streetview_coordinates_and_heading(*target_coordinates, API_KEY)

    # Check if the metadata retrieval was successful
    if result.get("status") == "success":
        try:
            # Extract coordinates from result
            closest_coords = result["closest_coordinates"]
            maps_coordinates = (closest_coords['lat'], closest_coords['lng'])
            
            # Calculate heading and distance
            heading, distance = calculate_heading_and_distance(
                maps_coordinates[0], maps_coordinates[1],
                target_coordinates[0], target_coordinates[1]
            )

            # Check if the distance is within the acceptable range
            if distance < max_distance:
                image_path = get_image(
                    target_coordinates[0], target_coordinates[1],
                    size, heading, pitch, fov, index
                )

                return image_path, distance

            else:

                return None, distance

        except KeyError as e:
            print(f"Missing expected key in response: {e}")
            return None, None
    else:
        # Handle errors returned by the metadata function
        print(f"Error retrieving metadata: {result.get('message', 'Unknown error')}")
        return None, None


